#!/usr/bin/env python3
"""
Простое оконное приложение на Tkinter для визуального конструирования
логических схем из блоков AND / OR / XOR / NOT, а также Input и Output.

Возможности:
* Добавление блоков из палитры.
* Перемещение блоков на рабочем поле.
* Соединение блоков (режим «Соединить»).
* Ввод значений на входных блоках (двойной клик меняет 0/1).
* Расчёт и отображение результата на всех выходных блоках.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


BLOCK_COLORS: Dict[str, str] = {
    "INPUT": "#f6b26b",
    "OUTPUT": "#93c47d",
    "AND": "#6fa8dc",
    "OR": "#d5a6bd",
    "XOR": "#ffd966",
    "NOT": "#b4a7d6",
    "NAND": "#76a5af",
    "NOR": "#c27ba0",
}


GATE_SPECS: Dict[str, Dict[str, object]] = {
    "AND": {"inputs": 2, "func": lambda values: all(values)},
    "OR": {"inputs": 2, "func": lambda values: any(values)},
    "XOR": {"inputs": 2, "func": lambda values: sum(values) % 2 == 1},
    "NOT": {"inputs": 1, "func": lambda values: not values[0]},
    "NAND": {"inputs": 2, "func": lambda values: not all(values)},
    "NOR": {"inputs": 2, "func": lambda values: not any(values)},
    "OUTPUT": {"inputs": 1, "func": lambda values: values[0]},
}


@dataclass
class Connection:
    source: "Block"
    target: "Block"
    line_id: int
    target_slot: Optional[int] = None

    def refresh(self) -> None:
        x1, y1 = self.source.output_anchor()
        x2, y2 = self.target.input_anchor(self.target_slot)
        self.source.app.canvas.coords(self.line_id, x1, y1, x2, y2)


@dataclass
class Block:
    app: "LogicBlocksApp"
    kind: str
    x: int = 40
    y: int = 40
    width: int = 100
    height: int = 50
    rect_id: Optional[int] = None
    text_id: Optional[int] = None
    value: Optional[bool] = False
    active: bool = False
    incoming: List[Connection] = field(default_factory=list)
    outgoing: List[Connection] = field(default_factory=list)
    connector_ids: List[int] = field(default_factory=list)
    slot_map: Dict[int, Connection] = field(default_factory=dict)
    indicator_id: Optional[int] = None  # ДОБАВЛЕНО: инициализация indicator_id

    def __post_init__(self) -> None:
        self.draw()

    def draw(self) -> None:
        canvas = self.app.canvas
        self.rect_id = canvas.create_rectangle(
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
            fill=BLOCK_COLORS.get(self.kind, "#cccccc"),
            outline="#3c3c3c",
            width=2,
        )
        self.text_id = canvas.create_text(
            self.x + self.width / 2,
            self.y + self.height / 2,
            text=self.label_text(),
            font=("Segoe UI", 11, "bold"),
        )
        self.bind_events(self.rect_id)
        self.bind_events(self.text_id)
        self.draw_connectors()
        self.update_style()

    def label_text(self) -> str:
        if self.kind == "INPUT":
            status = "ON" if self.active else "OFF"
            return f"IN {status}"
        if self.kind == "OUTPUT":
            return "OUT"
        return self.kind

    def bind_events(self, item_id: int) -> None:
        canvas = self.app.canvas
        canvas.tag_bind(item_id, "<Button-1>", self.on_click)
        canvas.tag_bind(item_id, "<B1-Motion>", self.on_drag)
        canvas.tag_bind(item_id, "<ButtonRelease-1>", self.on_release)
        canvas.tag_bind(item_id, "<Double-1>", self.on_double_click)
        canvas.tag_bind(item_id, "<Button-3>", self.on_right_click)

    def on_click(self, event: tk.Event) -> None:
        if self.app.mode == "connect":
            self.app.handle_connection_click(self)
            return
        self.app.select_block(self)
        self._drag_offset = (event.x - self.x, event.y - self.y)
        self._dragged = False

    def on_drag(self, event: tk.Event) -> None:
        if getattr(self, "_drag_offset", None) is None or self.app.mode == "connect":
            return
        self._dragged = True
        dx, dy = self._drag_offset
        self.move_to(event.x - dx, event.y - dy)

    def on_release(self, _event: tk.Event) -> None:
        if (
            self.app.mode != "connect"
            and self.kind == "INPUT"
            and not getattr(self, "_dragged", False)
        ):
            self.active = not self.active
            self.update_style()
            self.app.evaluate()
        self._drag_offset = None
        self._dragged = False

    def on_double_click(self, _event: tk.Event) -> None:
        if self.kind == "INPUT":
            self.active = True
            self.value = not bool(self.value)
            self.update_label()
            self.app.evaluate()
        elif self.kind == "OUTPUT":
            self.app.evaluate()

    def on_right_click(self, _event: tk.Event) -> None:
        if self.kind == "INPUT":
            self.active = not self.active
            self.update_style()
            self.app.evaluate()

    def move_to(self, new_x: int, new_y: int) -> None:
        self.x = max(10, new_x)
        self.y = max(10, new_y)
        canvas = self.app.canvas
        canvas.coords(
            self.rect_id,
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )
        canvas.coords(
            self.text_id,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )
        self.draw_connectors()
        self.update_indicator()
        # ОБНОВЛЯЕМ ВСЕ СОЕДИНЕНИЯ - И ВХОДЯЩИЕ И ИСХОДЯЩИЕ
        for conn in self.incoming + self.outgoing:
            conn.refresh()

    def can_output(self) -> bool:
        return True

    def max_inputs(self) -> int:
        if self.kind == "INPUT":
            return 0
        return int(GATE_SPECS.get(self.kind, {}).get("inputs", 2))

    def has_input_capacity(self) -> bool:
        return len(self.slot_map) < self.max_inputs()

    def input_anchor(self, slot_idx: Optional[int] = None) -> tuple[int, int]:
        positions = self.input_slot_positions()
        if slot_idx is not None and 0 <= slot_idx < len(positions):
            anchor_y = positions[slot_idx]
        else:
            anchor_y = positions[0] if positions else self.y + self.height / 2
        return (self.x - 14, anchor_y)

    def output_anchor(self) -> tuple[int, int]:
        return (self.x + self.width + 14, self.y + self.height / 2)

    def remove_connection(self, connection: Connection) -> None:
        if connection in self.incoming:
            self.incoming.remove(connection)
            self.release_input_slot(connection)
        if connection in self.outgoing:
            self.outgoing.remove(connection)

    def update_label(self) -> None:
        if self.text_id is not None:
            self.app.canvas.itemconfigure(self.text_id, text=self.label_text())

    def draw_connectors(self) -> None:
        canvas = self.app.canvas
        for cid in self.connector_ids:
            canvas.delete(cid)
        self.connector_ids.clear()

        connector_color = "#333333"
        for anchor_y in self.input_slot_positions():
            cid = canvas.create_line(
                self.x - 14,
                anchor_y,
                self.x,
                anchor_y,
                width=4,
                fill=connector_color,
                capstyle=tk.ROUND,
            )
            self.connector_ids.append(cid)

        anchor_y = self.y + self.height / 2
        cid = canvas.create_line(
            self.x + self.width,
            anchor_y,
            self.x + self.width + 14,
            anchor_y,
            width=4,
            fill=connector_color,
            capstyle=tk.ROUND,
        )
        self.connector_ids.append(cid)

    def input_slot_positions(self) -> List[float]:
        slots = self.max_inputs()
        if slots <= 0:
            return []
        return [
            self.y + self.height * (idx + 1) / (slots + 1)
            for idx in range(slots)
        ]

    def assign_input_slot(self, connection: Connection) -> Optional[int]:
        slots = self.max_inputs()
        if slots <= 0:
            return None
        for idx in range(slots):
            if idx not in self.slot_map:
                self.slot_map[idx] = connection
                return idx
        return None

    def release_input_slot(self, connection: Connection) -> None:
        for idx, conn in list(self.slot_map.items()):
            if conn == connection:
                del self.slot_map[idx]
                break

    def update_style(self) -> None:
        if self.rect_id is None:
            return
        fill = BLOCK_COLORS.get(self.kind, "#cccccc")
        if self.kind == "INPUT":
            fill = "#7ac943" if self.active else "#e06666"
        self.app.canvas.itemconfigure(self.rect_id, fill=fill)
        self.update_label()
        self.update_indicator()

    def update_indicator(self) -> None:
        canvas = self.app.canvas
        if self.kind != "OUTPUT":
            # ИСПРАВЛЕНО: проверяем существование indicator_id перед удалением
            if hasattr(self, 'indicator_id') and self.indicator_id:
                canvas.delete(self.indicator_id)
                self.indicator_id = None
            return
        
        radius = 9
        cx = self.x + self.width - 18
        cy = self.y + self.height - 14
        
        # ИСПРАВЛЕНО: создаем индикатор если его нет
        if self.indicator_id is None:
            self.indicator_id = canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline="",
                width=0,
            )
        else:
            canvas.coords(
                self.indicator_id,
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            )
        
        if self.value is None:
            color = "#b7b7b7"
        elif bool(self.value):
            color = "#ffb347"
        else:
            color = "#5b9bd5"
        canvas.itemconfigure(self.indicator_id, fill=color)


class LogicBlocksApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Логические блоки")
        self.root.geometry("1100x650")
        self.mode = "select"
        self.blocks: List[Block] = []
        self.connections: List[Connection] = []
        self.selected_block: Optional[Block] = None
        self.pending_source: Optional[Block] = None
        self.spawn_step = 80
        self.spawn_start = {
            "INPUT": (80, 80),
            "LOGIC": (320, 80),
            "OUTPUT": (560, 80),
        }
        self.spawn_counters = {
            "INPUT": 0,
            "LOGIC": 0,
            "OUTPUT": 0,
        }
        self.build_ui()

    def build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        palette = ttk.Frame(main_frame)
        palette.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            palette,
            text="Инструменты",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 4))

        for kind in ["INPUT", "AND", "OR", "XOR", "NOT", "NAND", "NOR", "OUTPUT"]:
            ttk.Button(
                palette,
                text=kind,
                command=lambda k=kind: self.create_block(k),
            ).pack(fill=tk.X, pady=2)

        ttk.Separator(palette, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(
            palette,
            text="Управление",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 4))

        ttk.Button(
            palette,
            text="Соединить блоки",
            command=self.toggle_connect_mode,
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            palette,
            text="Удалить выбранные",
            command=self.delete_selected,
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            palette,
            text="Сбросить схему",
            command=self.reset_workspace,
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(palette, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(
            palette,
            text="Информация",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 4))

        self.status_var = tk.StringVar(value="Совет: добавьте блоки и соедините их.")
        ttk.Label(
            palette,
            textvariable=self.status_var,
            wraplength=220,
        ).pack(fill=tk.X, pady=(8, 0))

        workspace = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        workspace.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_container = ttk.Frame(workspace)
        workspace.add(canvas_container, weight=3)

        self.canvas = tk.Canvas(
            canvas_container,
            background="#f8f8f8",
            highlightthickness=1,
            highlightbackground="#bcbcbc",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        truth_frame = ttk.Frame(workspace, padding=(10, 0, 0, 0))
        workspace.add(truth_frame, weight=1)

        ttk.Label(
            truth_frame,
            text="Таблица истинности",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 4))

        tree_container = ttk.Frame(truth_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        self.truth_tree = ttk.Treeview(
            tree_container,
            columns=("INFO",),
            show="headings",
            height=18,
        )
        self.truth_tree.heading("INFO", text="Информация")
        self.truth_tree.column("INFO", width=260, anchor=tk.CENTER, stretch=True)

        truth_vscroll = ttk.Scrollbar(
            tree_container,
            orient=tk.VERTICAL,
            command=self.truth_tree.yview,
        )
        truth_hscroll = ttk.Scrollbar(
            tree_container,
            orient=tk.HORIZONTAL,
            command=self.truth_tree.xview,
        )
        self.truth_tree.configure(
            yscrollcommand=truth_vscroll.set,
            xscrollcommand=truth_hscroll.set,
        )
        self.truth_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        truth_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        truth_hscroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_truth_table_widget()

    def spawn_category(self, kind: str) -> str:
        if kind == "INPUT":
            return "INPUT"
        if kind == "OUTPUT":
            return "OUTPUT"
        return "LOGIC"

    def next_spawn_position(self, kind: str) -> tuple[int, int]:
        category = self.spawn_category(kind)
        base_x, base_y = self.spawn_start[category]
        offset = self.spawn_counters[category] * self.spawn_step
        self.spawn_counters[category] += 1
        return base_x, base_y + offset

    def create_block(self, kind: str) -> None:
        x, y = self.next_spawn_position(kind)
        block = Block(self, kind, x=x, y=y)
        self.blocks.append(block)
        self.status_var.set(f"Создан блок {kind}. Перетащите его на холсте.")
        self.update_truth_table_widget()

    def select_block(self, block: Block) -> None:
        if self.selected_block == block:
            return
        if self.selected_block:
            self.canvas.itemconfigure(self.selected_block.rect_id, width=2)
        self.selected_block = block
        self.canvas.itemconfigure(block.rect_id, width=3)

    def toggle_connect_mode(self) -> None:
        if self.mode == "connect":
            self.mode = "select"
            self.pending_source = None
            self.status_var.set("Режим выбора. Двойной клик по INPUT меняет значение.")
        else:
            self.mode = "connect"
            self.pending_source = None
            self.status_var.set("Режим соединения: кликните источник, затем приёмник.")

    def handle_connection_click(self, block: Block) -> None:
        if self.pending_source is None:
            if block.max_inputs() == 0:
                self.pending_source = block
                self.status_var.set("Источник выбран. Теперь выберите блок-приёмник.")
            elif block.can_output():
                self.pending_source = block
                self.status_var.set("Источник выбран. Теперь выберите блок-приёмник.")
            else:
                messagebox.showinfo("Ошибка", "Этот блок не может быть источником.")
            return
        if block == self.pending_source:
            self.status_var.set("Нельзя соединить блок сам с собой.")
            return

        if not block.has_input_capacity():
            messagebox.showinfo("Ошибка", "У блока нет свободных входов.")
            return

        connection = Connection(
            source=self.pending_source,
            target=block,
            line_id=self.canvas.create_line(
                *self.pending_source.output_anchor(),
                *block.input_anchor(),
                width=2,
                fill="#555555",
                smooth=True,
            ),
        )
        slot = block.assign_input_slot(connection)
        if slot is None:
            self.canvas.delete(connection.line_id)
            messagebox.showinfo("Ошибка", "У блока нет свободных входов.")
            self.pending_source = None
            self.mode = "select"
            self.status_var.set("Нет свободных входов у выбранного блока.")
            return
        connection.target_slot = slot
        self.connections.append(connection)
        self.pending_source.outgoing.append(connection)
        block.incoming.append(connection)
        connection.refresh()
        self.status_var.set("Соединение добавлено.")
        self.update_truth_table_widget()
        self.toggle_connect_mode()

    def remove_block(self, block: Block) -> None:
        for connection in list(block.incoming):
            self.delete_connection(connection, silent=True)
        for connection in list(block.outgoing):
            self.delete_connection(connection, silent=True)
        if block.rect_id:
            self.canvas.delete(block.rect_id)
        if block.text_id:
            self.canvas.delete(block.text_id)
        for cid in block.connector_ids:
            self.canvas.delete(cid)
        if hasattr(block, 'indicator_id') and block.indicator_id:
            self.canvas.delete(block.indicator_id)
        self.blocks.remove(block)
        self.update_truth_table_widget()

    def delete_connection(self, connection: Connection, silent: bool = False) -> None:
        connection.source.remove_connection(connection)
        connection.target.remove_connection(connection)
        self.canvas.delete(connection.line_id)
        if connection in self.connections:
            self.connections.remove(connection)
        if not silent:
            self.update_truth_table_widget()

    def delete_selected(self) -> None:
        if not self.selected_block:
            self.status_var.set("Нет выбранного блока.")
            return
        self.remove_block(self.selected_block)
        self.selected_block = None
        self.status_var.set("Блок удалён.")

    def reset_workspace(self) -> None:
        for block in list(self.blocks):
            self.remove_block(block)
        self.connections.clear()
        self.selected_block = None
        self.pending_source = None
        self.mode = "select"
        for key in self.spawn_counters:
            self.spawn_counters[key] = 0
        self.update_truth_table_widget()
        self.status_var.set("Поле очищено.")

    def compute_all(self, input_override: Optional[Dict[int, bool]] = None) -> Dict[int, Optional[bool]]:
        override = input_override or {}
        visited: Dict[int, Optional[bool]] = {}

        def compute(block: Block, stack: Set[int]) -> Optional[bool]:
            block_id = id(block)
            if block_id in visited:
                return visited[block_id]
            if block_id in stack:
                raise ValueError("Обнаружен цикл соединений.")
            if block.kind == "INPUT":
                if not block.active:
                    visited[block_id] = None
                    return None
                result = override.get(block_id, bool(block.value))
                visited[block_id] = result
                return result
            stack.add(block_id)
            inputs: List[bool] = []
            for connection in block.incoming:
                source_val = compute(connection.source, stack)
                if source_val is None:
                    stack.remove(block_id)
                    visited[block_id] = None
                    return None
                inputs.append(source_val)
            required = block.max_inputs()
            if len(inputs) < required:
                visited[block_id] = None
                stack.remove(block_id)
                return None
            func: Optional[Callable[[List[bool]], bool]] = GATE_SPECS.get(block.kind, {}).get("func")  # type: ignore[assignment]
            if func is None:
                visited[block_id] = None
            else:
                visited[block_id] = bool(func(inputs))
            stack.remove(block_id)
            return visited[block_id]

        for block in self.blocks:
            compute(block, set())
        return visited

    def evaluate(self) -> None:
        try:
            results = self.compute_all()
            for block in self.blocks:
                block.value = results.get(id(block))
                block.update_label()
                block.update_indicator()  # ДОБАВЛЕНО: обновляем индикатор
            self.status_var.set("Расчёт завершён.")
            self.update_truth_table_widget()
        except ValueError as exc:
            messagebox.showerror("Ошибка схемы", str(exc))
            self.status_var.set("Ошибка: проверьте наличие циклов.")

    def update_truth_table_widget(self, alert: bool = False) -> None:
        input_blocks = [block for block in self.blocks if block.kind == "INPUT"]
        output_blocks = [block for block in self.blocks if block.kind == "OUTPUT"]

        if not hasattr(self, "truth_tree"):
            return

        active_inputs = [block for block in input_blocks if block.active]

        if not input_blocks or not output_blocks:
            if alert:
                message = (
                    "Добавьте хотя бы один блок INPUT."
                    if not input_blocks
                    else "Добавьте хотя бы один блок OUTPUT."
                )
                messagebox.showinfo("Таблица истинности", message)
            self.truth_tree.configure(columns=("INFO",))
            self.truth_tree.heading("INFO", text="Информация")
            self.truth_tree.column("INFO", width=260, anchor=tk.CENTER, stretch=True)
            for item in self.truth_tree.get_children():
                self.truth_tree.delete(item)
            placeholder = "Добавьте INPUT и OUTPUT."
            self.truth_tree.insert("", tk.END, values=(placeholder,))
            return

        if not active_inputs:
            self.truth_tree.configure(columns=("INFO",))
            self.truth_tree.heading("INFO", text="Информация")
            self.truth_tree.column("INFO", width=260, anchor=tk.CENTER, stretch=True)
            for item in self.truth_tree.get_children():
                self.truth_tree.delete(item)
            placeholder = "Активируйте входы (клик по блоку)."
            self.truth_tree.insert("", tk.END, values=(placeholder,))
            if alert:
                messagebox.showinfo("Таблица истинности", "Активируйте хотя бы один INPUT.")
            return

        max_inputs = 10
        if len(input_blocks) > max_inputs:
            if alert:
                messagebox.showwarning(
                    "Таблица истинности",
                    f"Слишком много входов ({len(input_blocks)}). Ограничение: {max_inputs}.",
                )
            self.status_var.set("Слишком много входов для таблицы.")
            self.truth_tree.configure(columns=("INFO",))
            self.truth_tree.heading("INFO", text="Информация")
            self.truth_tree.column("INFO", width=260, anchor=tk.CENTER, stretch=True)
            for item in self.truth_tree.get_children():
                self.truth_tree.delete(item)
            self.truth_tree.insert("", tk.END, values=("Ограничение: максимум 10 входов.",))
            return

        rows: List[List[str]] = []
        total_rows = 1 << len(active_inputs)

        try:
            for mask in range(total_rows):
                override = {
                    id(active_inputs[idx]): bool((mask >> idx) & 1)
                    for idx in range(len(active_inputs))
                }
                results = self.compute_all(override)
                row: List[str] = [
                    str(int(override[id(block)])) for block in active_inputs
                ]
                for block in output_blocks:
                    value = results.get(id(block))
                    row.append("-" if value is None else str(int(bool(value))))
                rows.append(row)
        except ValueError as exc:
            if alert:
                messagebox.showerror("Ошибка схемы", str(exc))
            self.status_var.set("Ошибка: проверьте наличие циклов.")
            return

        columns = (
            [f"IN{i + 1}" for i in range(len(active_inputs))]
            + [f"OUT{j + 1}" for j in range(len(output_blocks))]
        )
        self.truth_tree.configure(columns=columns)
        for col in columns:
            self.truth_tree.heading(col, text=col)
            self.truth_tree.column(col, width=90, anchor=tk.CENTER, stretch=True)

        for item in self.truth_tree.get_children():
            self.truth_tree.delete(item)

        for row in rows:
            self.truth_tree.insert("", tk.END, values=row)

        if alert:
            self.status_var.set("Таблица истинности обновлена.")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    LogicBlocksApp().run()