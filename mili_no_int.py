# -----------------------------------------
# Данные автомата (пример по твоей таблице)
# -----------------------------------------

# входы: 1, 2, 3  соответствует a1, a2, a3
inputs = [1, 2, 3]

# состояния: 1..9
states = list(range(1, 10))

# output[state][input] = y(t)
outputs = {
    1: {1:0, 2:1, 3:1},
    2: {1:1, 2:0, 3:0},
    3: {1:1, 2:0, 3:0},
    4: {1:0, 2:1, 3:1},
    5: {1:1, 2:1, 3:0},
    6: {1:0, 2:1, 3:1},
    7: {1:1, 2:1, 3:0},
    8: {1:1, 2:0, 3:0},
    9: {1:0, 2:1, 3:1},
}

# next_state[state][input] = s(t+1)
next_state = {
    1: {1:2, 2:4, 3:4},
    2: {1:1, 2:1, 3:5},
    3: {1:1, 2:6, 3:5},
    4: {1:8, 2:1, 3:1},
    5: {1:6, 2:4, 3:3},
    6: {1:8, 2:9, 3:6},
    7: {1:6, 2:1, 3:3},
    8: {1:4, 2:4, 3:7},
    9: {1:7, 2:9, 3:7},
}

# -----------------------------------------------------
# ЭТАП 1: построение строк вида (x, y, s) -> s_next
# -----------------------------------------------------

rows = []

for a in inputs:
    for s in states:
        y = outputs[s][a]
        s_next = next_state[s][a]
        rows.append({
            "x": a,
            "y": y,
            "s": s,
            "s_next": s_next
        })

# (служебно) выводим Bx / Byx, как раньше
print("=== Таблица (x, y, s) → s(t+1) ===")
for r in rows:
    print(f"x={r['x']}  y={r['y']}  s={r['s']}  ->  {r['s_next']}")

# -----------------------------------------------------
# ЭТАП 1: группировка по (x, y) — Bx / Byx (как у тебя)
# -----------------------------------------------------

primary_xy_sets = {}  # key = (x, y), value = список состояний

for r in rows:
    key = (r["x"], r["y"])
    primary_xy_sets.setdefault(key, []).append(r["s"])

print("\n=== Множества Bx / Byx (служебные) ===")
all_y_values = sorted({key[1] for key in primary_xy_sets.keys()})
all_x_values = sorted(inputs)

for y in all_y_values:
    print(f"\n--- y = {y} ---")
    for x in all_x_values:
        key = (x, y)
        if key in primary_xy_sets:
            print(f"Bx={x}, y={y}: {sorted(primary_xy_sets[key])}")

# -----------------------------------------------------
# НАСТОЯЩИЕ первичные классы Σ_1j:
#   состояния с одинаковой строкой выходов (y(a1), y(a2), y(a3))
# -----------------------------------------------------

def build_primary_classes(states, inputs, outputs):
    pattern_to_states = {}
    for s in states:
        pattern = tuple(outputs[s][a] for a in inputs)  # вектор выходов строки
        pattern_to_states.setdefault(pattern, []).append(s)
    # вернём список блоков-классов
    return list(pattern_to_states.values())

primary_classes = build_primary_classes(states, inputs, outputs)

print("\n=== Первичные классы Σ_1j (по строкам выходов) ===")
for j, block in enumerate(primary_classes, start=1):
    print(f"Σ1{j} = {sorted(block)}")

# -----------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЭТАПА 2
# -----------------------------------------------------
# -------- вспомогательные функции --------

def build_primary_classes(states, inputs, outputs):
    """Первичные классы Σ1j: состояния с одинаковой строкой выходов."""
    pattern_to_states = {}
    for s in states:
        pattern = tuple(outputs[s][a] for a in inputs)
        pattern_to_states.setdefault(pattern, []).append(s)
    return list(pattern_to_states.values())


def partition_to_state_map(partition):
    """Отображение: состояние -> номер класса (индекс блока)."""
    mapping = {}
    for idx, block in enumerate(partition):
        for s in block:
            mapping[s] = idx
    return mapping


def normalize_partition(partition):
    """Нормализуем разбиение для сравнения (игнорируем порядок классов)."""
    return sorted([tuple(sorted(block)) for block in partition])


def refine_partition(partition, inputs, next_state):
    """
    Один шаг уточнения разбиения.
    Новые классы добавляются в КОНЕЦ (их номера j будут последними).
    """
    state_to_class = partition_to_state_map(partition)
    new_partition = []
    extra_blocks = []  # сюда складываем новые классы, чтобы добавить их в конец

    for block in partition:
        # группируем состояния блока по "подписи" переходов
        signature_to_states = {}
        for s in block:
            signature = tuple(
                state_to_class[next_state[s][a]] for a in inputs
            )
            signature_to_states.setdefault(signature, []).append(s)

        subgroups = list(signature_to_states.values())

        if len(subgroups) == 1:
            # блок не делится
            new_partition.append(subgroups[0])
        else:
            # блок делится: первый подблок остаётся на месте,
            # остальные идут в extra_blocks => новые классы в конце
            subgroups.sort(key=lambda grp: min(grp))  # для устойчивости
            new_partition.append(subgroups[0])
            for sg in subgroups[1:]:
                extra_blocks.append(sg)

    new_partition.extend(extra_blocks)
    return new_partition


def print_classes(step_idx, partition):
    """Вывод классов Σ_{step_idx,j} без таблиц."""
    print(f"\n=== Классы Σ{step_idx}j ===")
    for j, block in enumerate(partition, start=1):
        print(f"Σ{step_idx}{j} = {sorted(block)}")


def print_step_tables(step_idx, partition, inputs, next_state):
    """
    Таблицы переходов для классов Σ_{step_idx,j}.
    В ячейке – имя класса, куда попадает состояние по входу.
    """
    # отображение: состояние -> номер класса
    state_to_class = partition_to_state_map(partition)

    # имена классов типа Σij
    class_name_by_index = {idx: f"Σ{step_idx}{j+1}"
                           for j, idx in enumerate(range(len(partition)))}

    print(f"\n--- Таблицы для шага {step_idx} ---")

    for j, block in enumerate(partition, start=1):
        name = f"Σ{step_idx}{j}"
        print(f"\nКласс {name} = {sorted(block)}")
        print("s(t) | " + "  ".join(f"a{a}" for a in inputs))
        print("-" * 30)
        for s in sorted(block):
            targets = []
            for a in inputs:
                s_next = next_state[s][a]
                cls_idx = state_to_class[s_next]
                targets.append(class_name_by_index[cls_idx])
            print(f"{s:3} | " + "  ".join(f"{t:>4}" for t in targets))


# -------- ЭТАП 2: цикл разбиения --------

# первичные классы Σ1j по строкам выходов
current_partition = build_primary_classes(states, inputs, outputs)

step = 1

while True:
    # 1) Вывод классов Σ_step,j
    print_classes(step, current_partition)

    # 2) Вывод таблиц
    print_step_tables(step, current_partition, inputs, next_state)

    # 3) Уточнение
    new_partition = refine_partition(current_partition, inputs, next_state)

    # Проверка на стабилизацию:
    if normalize_partition(new_partition) == normalize_partition(current_partition):
        print(f"\nШаг {step+1}: разбиение НЕ меняется => это последний шаг.")
        # ВАЖНО: нужно вывести Σ(step+1), хотя оно совпадает с Σ(step)
        print_classes(step+1, new_partition)
        print_step_tables(step+1, new_partition, inputs, next_state)
        final_partition = new_partition
        break

    # иначе продолжаем
    current_partition = new_partition
    step += 1

# итоговые классы (можно назвать Σ_final,j или просто перечислить)
print("\n=== Итоговое разбиение классов эквивалентности ===")
for j, block in enumerate(current_partition, start=1):
    print(f"Класс {j}: {sorted(block)}")

# ------------------------
# ЭТАП 3: минимальный автомат
# ------------------------

print("\n=== ЭТАП 3: Построение минимального автомата ===")

# final_partition получен на предыдущем шаге
P_equiv = final_partition

print("\nЭквивалентные классы Pэкв:")
for j, block in enumerate(P_equiv, start=1):
    print(f"Σ{j} = {sorted(block)}")

# ------------------------
# Новые состояния
# ------------------------

print("\nНовые состояния автомата:")
new_states = {}   # S(j) -> representative state
index_to_block = {}

for j, block in enumerate(P_equiv, start=1):
    rep = sorted(block)[0]   # берём первый как представителя
    new_states[j] = rep
    index_to_block[j] = block
    print(f"S({j}) = {sorted(block)}   (представитель = {rep})")

# ------------------------
# Таблица минимального автомата
# ------------------------

print("\n=== Минимизированный автомат ===")

print("S(t) | " + "  ".join(f"a{a}" for a in inputs) +
      " || " + "  ".join(f"a{a}" for a in inputs))

print("     | " + "  ".join(" y " for _ in inputs) +
      " || " + "  ".join("s(t+1)" for _ in inputs))

print("-" * 60)

# строим таблицу
state_to_class = partition_to_state_map(P_equiv)

for j in sorted(new_states.keys()):
    rep = new_states[j]

    row_y = [outputs[rep][a] for a in inputs]
    row_s = [state_to_class[next_state[rep][a]]+1 for a in inputs]

    print(f"S({j}) | " +
          "  ".join(str(v) for v in row_y) +
          " || " +
          "  ".join(f"S({cls})" for cls in row_s))