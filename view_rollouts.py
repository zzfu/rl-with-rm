"""
Rollout Viewer - Gradio UI for exploring GRPO training rollouts.

Usage:
    python view_rollouts.py [path/to/rollouts.db]

If no path provided, opens a file selector.
"""

import argparse
import os
import sqlite3
from typing import Optional

import gradio as gr
import pandas as pd


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection."""
    return sqlite3.connect(db_path)


def get_available_steps(db_path: str, table: str = "rollouts") -> list[int]:
    """Get list of available steps in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(f"SELECT DISTINCT step FROM {table} ORDER BY step")
    steps = [row[0] for row in cursor.fetchall()]
    conn.close()
    return steps


def get_step_stats(db_path: str, table: str = "rollouts") -> pd.DataFrame:
    """Get statistics for each step."""
    conn = get_db_connection(db_path)

    if table == "rollouts":
        query = """
            SELECT
                step,
                epoch,
                COUNT(*) as num_rollouts,
                COUNT(DISTINCT prompt_index) as num_prompts,
                AVG(reward) as avg_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                AVG(advantage) as avg_advantage
            FROM rollouts
            GROUP BY step
            ORDER BY step
        """
    else:
        query = """
            SELECT
                step,
                epoch,
                COUNT(*) as num_rollouts,
                COUNT(DISTINCT prompt_index) as num_prompts,
                AVG(reward) as avg_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward
            FROM eval_rollouts
            GROUP BY step
            ORDER BY step
        """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_rollouts_for_step(
    db_path: str,
    step: int,
    table: str = "rollouts",
    limit: int = 100
) -> pd.DataFrame:
    """Get rollouts for a specific step."""
    conn = get_db_connection(db_path)

    if table == "rollouts":
        query = """
            SELECT
                prompt_index,
                rollout_index,
                prompt,
                completion,
                reward,
                advantage
            FROM rollouts
            WHERE step = ?
            ORDER BY prompt_index, rollout_index
            LIMIT ?
        """
    else:
        query = """
            SELECT
                prompt_index,
                prompt,
                completion,
                reward
            FROM eval_rollouts
            WHERE step = ?
            ORDER BY prompt_index
            LIMIT ?
        """

    df = pd.read_sql_query(query, conn, params=(step, limit))
    conn.close()
    return df


def get_prompt_group(
    db_path: str,
    step: int,
    prompt_index: int,
) -> pd.DataFrame:
    """Get all rollouts for a specific prompt at a step."""
    conn = get_db_connection(db_path)
    query = """
        SELECT
            rollout_index,
            completion,
            reward,
            advantage
        FROM rollouts
        WHERE step = ? AND prompt_index = ?
        ORDER BY rollout_index
    """
    df = pd.read_sql_query(query, conn, params=(step, prompt_index))
    conn.close()
    return df


def get_prompt_count(db_path: str, step: int) -> int:
    """Get the number of distinct prompts at a step."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT COUNT(DISTINCT prompt_index) FROM rollouts WHERE step = ?",
        (step,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_prompt_at_index(db_path: str, step: int, prompt_index: int) -> tuple[str, pd.DataFrame]:
    """Get prompt text and all completions for a specific prompt at a step."""
    conn = get_db_connection(db_path)

    # Get prompt text
    cursor = conn.execute(
        "SELECT prompt FROM rollouts WHERE step = ? AND prompt_index = ? LIMIT 1",
        (step, prompt_index)
    )
    row = cursor.fetchone()
    prompt_text = row[0] if row else ""

    # Get all completions
    query = """
        SELECT
            rollout_index,
            completion,
            reward,
            advantage
        FROM rollouts
        WHERE step = ? AND prompt_index = ?
        ORDER BY reward DESC
    """
    df = pd.read_sql_query(query, conn, params=(step, prompt_index))
    conn.close()

    return prompt_text, df


def format_completion_display(row: pd.Series, show_full: bool = False) -> str:
    """Format a rollout for display."""
    completion = row["completion"]
    reward = row["reward"]

    # Truncate if needed
    if not show_full and len(completion) > 500:
        completion = completion[:500] + "..."

    # Format advantage if present
    adv_str = ""
    if "advantage" in row and pd.notna(row["advantage"]):
        adv_str = f" | Advantage: {row['advantage']:.4f}"

    return f"**Reward: {reward:.4f}**{adv_str}\n\n{completion}"


def render_prompt_html(prompt_text: str) -> str:
    """Render a chat prompt as nicely formatted HTML with role labels."""
    import html
    import re

    if not prompt_text:
        return "<p style='color: #888;'>No prompt loaded.</p>"

    # Parse Qwen chat template format: <|im_start|>role\ncontent<|im_end|>
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt_text, re.DOTALL)

    if not matches:
        # Fallback: just escape and display raw
        return f"<pre style='white-space: pre-wrap; color: #e0e0e0;'>{html.escape(prompt_text)}</pre>"

    turns_html = []
    for role, content in matches:
        content = content.strip()
        escaped_content = html.escape(content).replace("\n", "<br>")

        # Style based on role
        if role == "user":
            role_color = "#60a5fa"  # Blue
            role_label = "User"
            bg_color = "#1e3a5f"
        elif role == "assistant":
            role_color = "#4ade80"  # Green
            role_label = "Assistant"
            bg_color = "#1a3d1a"
        elif role == "system":
            role_color = "#a78bfa"  # Purple
            role_label = "System"
            bg_color = "#2d2640"
        else:
            role_color = "#888"
            role_label = role.capitalize()
            bg_color = "#2a2a2a"

        turn_html = f"""
        <div style="
            margin: 8px 0;
            padding: 10px 12px;
            border-radius: 6px;
            background: {bg_color};
            border-left: 3px solid {role_color};
        ">
            <div style="
                font-weight: bold;
                color: {role_color};
                margin-bottom: 6px;
                font-size: 12px;
                text-transform: uppercase;
            ">{role_label}</div>
            <div style="
                color: #e0e0e0;
                font-size: 14px;
                line-height: 1.5;
            ">{escaped_content}</div>
        </div>
        """
        turns_html.append(turn_html)

    return f"<div>{''.join(turns_html)}</div>"


def render_comparison_cards(df: pd.DataFrame, selected_indices: list[int]) -> str:
    """Render HTML cards for selected completions."""
    import html

    if df.empty or not selected_indices:
        return "<p style='color: #888;'>Select completions from the sidebar to compare.</p>"

    # Filter to selected rollouts
    selected_df = df[df["rollout_index"].isin(selected_indices)]
    if selected_df.empty:
        return "<p style='color: #888;'>No matching completions found.</p>"

    # Get min/max rewards for color scaling
    all_rewards = df["reward"].tolist()
    min_r, max_r = min(all_rewards), max(all_rewards)
    reward_range = max_r - min_r if max_r != min_r else 1.0

    cards_html = []
    for _, row in selected_df.iterrows():
        rollout_idx = int(row["rollout_index"])
        reward = row["reward"]
        advantage = row["advantage"]
        completion = row["completion"]

        # Color based on reward (green=high, red=low)
        normalized = (reward - min_r) / reward_range
        if normalized > 0.5:
            # Green gradient
            green_intensity = int(100 + 155 * (normalized - 0.5) * 2)
            border_color = f"rgb(50, {green_intensity}, 50)"
        else:
            # Red gradient
            red_intensity = int(100 + 155 * (0.5 - normalized) * 2)
            border_color = f"rgb({red_intensity}, 50, 50)"

        # Format advantage
        adv_str = f"{advantage:+.4f}" if pd.notna(advantage) else "N/A"

        # Escape HTML in completion
        escaped_completion = html.escape(completion)
        # Convert newlines to <br> for display
        escaped_completion = escaped_completion.replace("\n", "<br>")

        card = f"""
        <div style="
            border: 3px solid {border_color};
            border-radius: 8px;
            padding: 12px;
            margin: 8px;
            background: #1a1a1a;
            min-width: 300px;
            max-width: 600px;
            flex: 1;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding-bottom: 8px;
                border-bottom: 1px solid #333;
            ">
                <span style="font-weight: bold; color: #fff;">#{rollout_idx}</span>
                <span style="color: #4ade80;">R: {reward:.4f}</span>
                <span style="color: #60a5fa;">Adv: {adv_str}</span>
            </div>
            <div style="
                max-height: 400px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 13px;
                white-space: pre-wrap;
                word-break: break-word;
                color: #e0e0e0;
            ">{escaped_completion}</div>
        </div>
        """
        cards_html.append(card)

    return f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: flex-start;
    ">
        {''.join(cards_html)}
    </div>
    """


def create_app(initial_db_path: Optional[str] = None):
    """Create the Gradio app."""

    with gr.Blocks(title="Rollout Viewer") as app:
        gr.Markdown("# 🎲 GRPO Rollout Viewer")
        gr.Markdown("Explore training and evaluation rollouts from GRPO training.")

        # Database selection
        with gr.Row():
            db_path_input = gr.Textbox(
                label="Database Path",
                placeholder="./rollouts/run_name/rollouts.db",
                value=initial_db_path or "",
                scale=4,
            )
            load_btn = gr.Button("Load", variant="primary", scale=1)

        db_status = gr.Markdown("")

        with gr.Tabs() as tabs:
            # Compare tab (primary use case)
            with gr.Tab("🔄 Compare"):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1, min_width=220):
                        compare_step = gr.Dropdown(
                            label="Step",
                            choices=[],
                            interactive=True,
                        )
                        gr.Markdown("---")
                        compare_prompt_count = gr.Markdown("Prompt 0 of 0")
                        with gr.Row():
                            compare_prev_btn = gr.Button("◀", size="sm", scale=1)
                            compare_prompt_idx = gr.Number(
                                label="Index",
                                value=0,
                                precision=0,
                                minimum=0,
                                scale=2,
                            )
                            compare_next_btn = gr.Button("▶", size="sm", scale=1)
                        gr.Markdown("---")
                        compare_checkboxes = gr.CheckboxGroup(
                            label="Completions",
                            choices=[],
                            value=[],
                            interactive=True,
                        )
                        with gr.Row():
                            compare_best2_btn = gr.Button("Best 2", size="sm")
                            compare_all_btn = gr.Button("All", size="sm")
                            compare_clear_btn = gr.Button("Clear", size="sm")

                    # Main area
                    with gr.Column(scale=4):
                        gr.Markdown("### Prompt")
                        compare_prompt_display = gr.HTML(
                            "<p style='color: #888;'>Select a step to begin.</p>",
                        )
                        gr.Markdown("### Completions")
                        compare_cards = gr.HTML(
                            "<p style='color: #888;'>Select completions from the sidebar to compare.</p>"
                        )

            # Overview tab
            with gr.Tab("📊 Overview"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Rollouts")
                        train_stats_df = gr.Dataframe(
                            label="Steps Overview",
                            headers=["step", "epoch", "num_rollouts", "num_prompts",
                                    "avg_reward", "min_reward", "max_reward", "avg_advantage"],
                        )
                    with gr.Column():
                        gr.Markdown("### Eval Rollouts")
                        eval_stats_df = gr.Dataframe(
                            label="Steps Overview",
                            headers=["step", "epoch", "num_rollouts", "num_prompts",
                                    "avg_reward", "min_reward", "max_reward"],
                        )

            # Training rollouts tab
            with gr.Tab("🎯 Training Rollouts"):
                with gr.Row():
                    train_step_dropdown = gr.Dropdown(
                        label="Step",
                        choices=[],
                        interactive=True,
                        scale=2,
                    )
                    train_refresh_btn = gr.Button("Refresh", scale=1)

                train_rollouts_df = gr.Dataframe(
                    label="Rollouts",
                    headers=["prompt_index", "rollout_index", "prompt", "completion", "reward", "advantage"],
                    wrap=True,
                    max_height=500,
                )

            # Eval rollouts tab
            with gr.Tab("📈 Eval Rollouts"):
                with gr.Row():
                    eval_step_dropdown = gr.Dropdown(
                        label="Step",
                        choices=[],
                        interactive=True,
                        scale=2,
                    )
                    eval_refresh_btn = gr.Button("Refresh", scale=1)

                eval_rollouts_df = gr.Dataframe(
                    label="Eval Rollouts",
                    headers=["prompt_index", "prompt", "completion", "reward"],
                    wrap=True,
                    max_height=500,
                )

            # Single rollout viewer
            with gr.Tab("🔍 Rollout Details"):
                gr.Markdown("### View Full Rollout")
                with gr.Row():
                    detail_table = gr.Radio(
                        choices=["rollouts", "eval_rollouts"],
                        value="rollouts",
                        label="Table",
                    )
                    detail_step = gr.Number(label="Step", precision=0)
                    detail_prompt_idx = gr.Number(label="Prompt Index", precision=0)
                    detail_rollout_idx = gr.Number(label="Rollout Index (training only)", precision=0)
                    detail_load_btn = gr.Button("Load", variant="primary")

                detail_prompt = gr.Textbox(label="Prompt", lines=5, interactive=False)
                detail_completion = gr.Textbox(label="Completion", lines=15, interactive=False)
                with gr.Row():
                    detail_reward = gr.Number(label="Reward", interactive=False)
                    detail_advantage = gr.Number(label="Advantage", interactive=False)

        # Event handlers
        def load_database(db_path: str):
            """Load database and return stats."""
            if not db_path or not os.path.exists(db_path):
                return (
                    f"❌ Database not found: {db_path}",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                )

            try:
                train_stats = get_step_stats(db_path, "rollouts")
                eval_stats = get_step_stats(db_path, "eval_rollouts")

                train_steps = get_available_steps(db_path, "rollouts")
                eval_steps = get_available_steps(db_path, "eval_rollouts")

                status = f"✅ Loaded: {db_path}\n\n"
                status += f"- Training rollouts: {len(train_steps)} steps\n"
                status += f"- Eval rollouts: {len(eval_steps)} steps"

                train_choices = [str(s) for s in train_steps]
                eval_choices = [str(s) for s in eval_steps]

                return (
                    status,
                    train_stats,
                    eval_stats,
                    gr.update(choices=train_choices, value=train_choices[0] if train_choices else None),
                    gr.update(choices=eval_choices, value=eval_choices[0] if eval_choices else None),
                    gr.update(choices=train_choices, value=train_choices[0] if train_choices else None),
                )
            except Exception as e:
                return (
                    f"❌ Error loading database: {e}",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                    gr.update(choices=[]),
                )

        def load_train_rollouts(step: str, db_path: str):
            """Load training rollouts for a step."""
            if not step or not db_path:
                return pd.DataFrame()
            try:
                return get_rollouts_for_step(db_path, int(step), "rollouts", limit=200)
            except Exception as e:
                return pd.DataFrame({"error": [str(e)]})

        def load_eval_rollouts(step: str, db_path: str):
            """Load eval rollouts for a step."""
            if not step or not db_path:
                return pd.DataFrame()
            try:
                return get_rollouts_for_step(db_path, int(step), "eval_rollouts", limit=200)
            except Exception as e:
                return pd.DataFrame({"error": [str(e)]})

        # Compare tab state (stored in a mutable container to share between handlers)
        compare_state = {"df": pd.DataFrame()}

        def on_compare_step_change(step: str, db_path: str):
            """Handle step change in compare tab."""
            if not step or not db_path:
                return (
                    "Prompt 0 of 0",
                    0,
                    gr.update(choices=[], value=[]),
                    "<p style='color: #888;'>Select a step to begin.</p>",
                    "<p style='color: #888;'>Select completions from the sidebar to compare.</p>",
                )

            try:
                step_int = int(step)
                count = get_prompt_count(db_path, step_int)
                if count == 0:
                    return (
                        "Prompt 0 of 0",
                        0,
                        gr.update(choices=[], value=[]),
                        "<p style='color: #888;'>No prompts found for this step.</p>",
                        "<p style='color: #888;'>No prompts found.</p>",
                    )

                # Load first prompt
                prompt_text, df = get_prompt_at_index(db_path, step_int, 0)
                compare_state["df"] = df

                # Build checkbox choices (sorted by reward descending)
                choices = [f"#{int(r['rollout_index'])} (r={r['reward']:.3f})" for _, r in df.iterrows()]

                return (
                    f"Prompt 1 of {count}",
                    0,
                    gr.update(choices=choices, value=choices[:2] if len(choices) >= 2 else choices),
                    render_prompt_html(prompt_text),
                    render_comparison_cards(df, list(df["rollout_index"][:2])),
                )
            except Exception as e:
                return (
                    "Prompt 0 of 0",
                    0,
                    gr.update(choices=[], value=[]),
                    f"<p style='color: #f87171;'>Error: {e}</p>",
                    "<p style='color: #888;'>Error loading data.</p>",
                )

        def on_compare_prompt_change(step: str, prompt_idx: float, db_path: str):
            """Handle prompt index change in compare tab."""
            if not step or prompt_idx is None or not db_path:
                return (
                    "Prompt 0 of 0",
                    gr.update(choices=[], value=[]),
                    "<p style='color: #888;'>Select a step to begin.</p>",
                    "<p style='color: #888;'>Select completions from the sidebar to compare.</p>",
                )

            try:
                step_int = int(step)
                prompt_idx_int = int(prompt_idx)
                count = get_prompt_count(db_path, step_int)

                # Clamp to valid range
                prompt_idx_int = max(0, min(prompt_idx_int, count - 1))

                prompt_text, df = get_prompt_at_index(db_path, step_int, prompt_idx_int)
                compare_state["df"] = df

                # Build checkbox choices
                choices = [f"#{int(r['rollout_index'])} (r={r['reward']:.3f})" for _, r in df.iterrows()]

                return (
                    f"Prompt {prompt_idx_int + 1} of {count}",
                    gr.update(choices=choices, value=choices[:2] if len(choices) >= 2 else choices),
                    render_prompt_html(prompt_text),
                    render_comparison_cards(df, list(df["rollout_index"][:2])),
                )
            except Exception as e:
                return (
                    "Prompt 0 of 0",
                    gr.update(choices=[], value=[]),
                    f"<p style='color: #f87171;'>Error: {e}</p>",
                    "<p style='color: #888;'>Error loading data.</p>",
                )

        def on_compare_prev(step: str, prompt_idx: float, db_path: str):
            """Go to previous prompt."""
            new_idx = max(0, int(prompt_idx or 0) - 1)
            return (new_idx,) + on_compare_prompt_change(step, new_idx, db_path)

        def on_compare_next(step: str, prompt_idx: float, db_path: str):
            """Go to next prompt."""
            if not step or not db_path:
                return (0,) + on_compare_prompt_change(step, 0, db_path)
            count = get_prompt_count(db_path, int(step))
            new_idx = min(count - 1, int(prompt_idx or 0) + 1)
            return (new_idx,) + on_compare_prompt_change(step, new_idx, db_path)

        def on_compare_checkbox_change(selected: list[str]):
            """Handle checkbox selection change."""
            df = compare_state["df"]
            if df.empty:
                return "<p style='color: #888;'>No data loaded.</p>"

            # Parse selected indices from checkbox labels like "#0 (r=0.123)"
            selected_indices = []
            for s in selected:
                try:
                    idx = int(s.split("#")[1].split(" ")[0])
                    selected_indices.append(idx)
                except (IndexError, ValueError):
                    pass

            return render_comparison_cards(df, selected_indices)

        def on_compare_select_best2():
            """Select top 2 by reward."""
            df = compare_state["df"]
            if df.empty:
                return gr.update(value=[])
            choices = [f"#{int(r['rollout_index'])} (r={r['reward']:.3f})" for _, r in df.iterrows()]
            return gr.update(value=choices[:2])

        def on_compare_select_all():
            """Select all completions."""
            df = compare_state["df"]
            if df.empty:
                return gr.update(value=[])
            choices = [f"#{int(r['rollout_index'])} (r={r['reward']:.3f})" for _, r in df.iterrows()]
            return gr.update(value=choices)

        def on_compare_clear():
            """Clear all selections."""
            return gr.update(value=[])

        def load_rollout_detail(table: str, step: int, prompt_idx: int, rollout_idx: int, db_path: str):
            """Load a single rollout's full details."""
            if step is None or prompt_idx is None or not db_path:
                return "", "", None, None

            try:
                conn = get_db_connection(db_path)

                if table == "rollouts":
                    cursor = conn.execute(
                        """SELECT prompt, completion, reward, advantage
                           FROM rollouts
                           WHERE step = ? AND prompt_index = ? AND rollout_index = ?""",
                        (int(step), int(prompt_idx), int(rollout_idx or 0))
                    )
                else:
                    cursor = conn.execute(
                        """SELECT prompt, completion, reward, NULL as advantage
                           FROM eval_rollouts
                           WHERE step = ? AND prompt_index = ?""",
                        (int(step), int(prompt_idx))
                    )

                row = cursor.fetchone()
                conn.close()

                if row:
                    return row[0], row[1], row[2], row[3]
                return "Not found", "", None, None
            except Exception as e:
                return f"Error: {e}", "", None, None

        # Wire up events
        load_btn.click(
            load_database,
            inputs=[db_path_input],
            outputs=[db_status, train_stats_df, eval_stats_df,
                    train_step_dropdown, eval_step_dropdown, compare_step],
        )

        train_step_dropdown.change(
            load_train_rollouts,
            inputs=[train_step_dropdown, db_path_input],
            outputs=[train_rollouts_df],
        )

        train_refresh_btn.click(
            load_train_rollouts,
            inputs=[train_step_dropdown, db_path_input],
            outputs=[train_rollouts_df],
        )

        eval_step_dropdown.change(
            load_eval_rollouts,
            inputs=[eval_step_dropdown, db_path_input],
            outputs=[eval_rollouts_df],
        )

        eval_refresh_btn.click(
            load_eval_rollouts,
            inputs=[eval_step_dropdown, db_path_input],
            outputs=[eval_rollouts_df],
        )

        detail_load_btn.click(
            load_rollout_detail,
            inputs=[detail_table, detail_step, detail_prompt_idx, detail_rollout_idx, db_path_input],
            outputs=[detail_prompt, detail_completion, detail_reward, detail_advantage],
        )

        # Compare tab events
        compare_step.change(
            on_compare_step_change,
            inputs=[compare_step, db_path_input],
            outputs=[compare_prompt_count, compare_prompt_idx, compare_checkboxes,
                    compare_prompt_display, compare_cards],
        )

        compare_prompt_idx.submit(
            on_compare_prompt_change,
            inputs=[compare_step, compare_prompt_idx, db_path_input],
            outputs=[compare_prompt_count, compare_checkboxes,
                    compare_prompt_display, compare_cards],
        )

        compare_prev_btn.click(
            on_compare_prev,
            inputs=[compare_step, compare_prompt_idx, db_path_input],
            outputs=[compare_prompt_idx, compare_prompt_count, compare_checkboxes,
                    compare_prompt_display, compare_cards],
        )

        compare_next_btn.click(
            on_compare_next,
            inputs=[compare_step, compare_prompt_idx, db_path_input],
            outputs=[compare_prompt_idx, compare_prompt_count, compare_checkboxes,
                    compare_prompt_display, compare_cards],
        )

        compare_checkboxes.change(
            on_compare_checkbox_change,
            inputs=[compare_checkboxes],
            outputs=[compare_cards],
        )

        compare_best2_btn.click(
            on_compare_select_best2,
            outputs=[compare_checkboxes],
        )

        compare_all_btn.click(
            on_compare_select_all,
            outputs=[compare_checkboxes],
        )

        compare_clear_btn.click(
            on_compare_clear,
            outputs=[compare_checkboxes],
        )

        # Auto-load on startup if path provided
        if initial_db_path:
            app.load(
                load_database,
                inputs=[db_path_input],
                outputs=[db_status, train_stats_df, eval_stats_df,
                        train_step_dropdown, eval_step_dropdown, compare_step],
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="View GRPO training rollouts")
    parser.add_argument(
        "db_path",
        nargs="?",
        default=None,
        help="Path to rollouts.db file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the server on (default: 7861)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    args = parser.parse_args()

    app = create_app(args.db_path)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
