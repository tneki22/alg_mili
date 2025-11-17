import streamlit as st
import pandas as pd

# ==========================================================
# МИНИМИЗАЦИЯ АВТОМАТА МИЛИ (итерационный алгоритм из методички)
# ----------------------------------------------------------
# Этот файл содержит:
#   1) Чистые функции для вычисления минимального автомата:
#      - build_primary_classes
#      - refine_partition
#      - run_minimization
#   2) Интерфейс на Streamlit для ввода таблиц и просмотра результата.
#
# Основные структуры данных (все – обычные Python-словари и списки):
#
#   inputs:      список входов      [1, 2, ..., m]
#   states:      список состояний   [1, 2, ..., n]
#   outputs:     словарь словарей   outputs[state][input] = выход y(t)
#   next_state:  словарь словарей   next_state[state][input] = s(t+1)
#
# Алгоритм минимизации:
#   Этап 1. Строим первичное разбиение Σ1j по строкам выходов (build_primary_classes).
#   Этап 2. Уточняем разбиение: на каждом шаге разбиваем классы по классам
#           следующих состояний (refine_partition) до стабилизации.
#   Этап 3. Каждому финальному классу Σj ставим в соответствие новое состояние S(j)
#           и строим таблицу минимального автомата (run_minimization).
# ==========================================================


# ----------- Вспомогательные функции для алгоритма -------------------

def build_primary_classes(states, inputs, outputs):
    """
    Строит первичные классы Σ1j.

    Идея:
    - для каждого состояния s берём строку выходов (y(a1), y(a2), ..., y(am));
    - состояния с одинаковой строкой выходов попадают в один и тот же класс.

    Параметры:
      states  – список номеров состояний (например, [1, 2, ..., n])
      inputs  – список входов (например, [1, 2, ..., m])
      outputs – словарь словарей: outputs[s][a] = y(t)

    Возвращает:
      список блоков (классов эквивалентности на первом шаге), например:
        [[1, 4, 6, 9], [2, 3, 8], [5, 7]]
    """
    pattern_to_states = {}
    for s in states:
        # "Подпись" состояния – вектор выходов по всем входам
        pattern = tuple(outputs[s][a] for a in inputs)
        pattern_to_states.setdefault(pattern, []).append(s)
    # Значения словаря – и есть блоки разбиения
    return list(pattern_to_states.values())


def partition_to_state_map(partition):
    """
    Строит отображение (словарь):
        state -> индекс блока (класса) в текущем разбиении.

    Пример:
        partition = [[1, 4], [2, 8], [5, 7], [3], [6], [9]]
        вернём:
        {
          1: 0, 4: 0,
          2: 1, 8: 1,
          5: 2, 7: 2,
          3: 3,
          6: 4,
          9: 5
        }
    """
    mapping = {}
    for idx, block in enumerate(partition):
        for s in block:
            mapping[s] = idx
    return mapping


def normalize_partition(partition):
    """
    Нормализует разбиение, чтобы можно было сравнивать его независимо
    от порядка элементов внутри блоков и порядка самих блоков.

    Идея: сортируем элементы в каждом блоке и сортируем список блоков.
    """
    return sorted([tuple(sorted(block)) for block in partition])


def refine_partition(partition, inputs, next_state):
    """
    Один шаг уточнения разбиения (Σi -> Σ(i+1)).

    Вход:
      partition  – текущее разбиение (список блоков состояний)
      inputs     – список входов
      next_state – словарь словарей: next_state[s][a] = s(t+1)

    Идея:
      - внутри каждого блока Σij группируем состояния по "подписи" переходов:
            signature(s) = ( class(next_state[s, a1]),
                              class(next_state[s, a2]),
                              ... )
      - если блок не делится, оставляем его как есть;
      - если делится, первый подблок остаётся на месте, остальные добавляем в конец,
        чтобы новые классы имели новые номера j.

    Возвращает:
      новое (уточнённое) разбиение.
    """
    state_to_class = partition_to_state_map(partition)
    new_partition = []
    extra_blocks = []  # сюда попадают новые блоки (классы), которые образовались в результате дробления

    for block in partition:
        # Группируем состояния блока по "подписи" переходов
        signature_to_states = {}
        for s in block:
            signature = tuple(
                state_to_class[next_state[s][a]] for a in inputs
            )
            signature_to_states.setdefault(signature, []).append(s)

        subgroups = list(signature_to_states.values())

        if len(subgroups) == 1:
            # Блок не делится – остаётся как есть
            new_partition.append(subgroups[0])
        else:
            # Блок делится на несколько подблоков:
            # - первый подблок остаётся на месте (тот же номер класса),
            # - остальные считаются "новыми" и добавляются в конец разбиения.
            subgroups.sort(key=lambda grp: min(grp))  # сортируем для устойчивости
            new_partition.append(subgroups[0])
            for sg in subgroups[1:]:
                extra_blocks.append(sg)

    # Все новые классы добавляем в конец
    new_partition.extend(extra_blocks)
    return new_partition


def run_minimization(inputs, states, outputs, next_state):
    """
    Запускает все этапы минимизации автомата Мили и возвращает
    развёрнутый текстовый отчёт.

    Параметры:
      inputs      – список входов (номеров столбцов таблицы)
      states      – список состояний
      outputs     – outputs[s][a]  = y(t)
      next_state  – next_state[s][a] = s(t+1)

    Этапы:
        1) строим первичные классы Σ1j (по строкам выходов);
        2) итеративно уточняем разбиение до стабилизации Σ(i+1) == Σi;
        3) по финальному разбиению Pэкв строим минимальный автомат Ã:
           - каждое Σj -> состояние S(j),
           - выходы/переходы берём из представителя класса.
    """

    text = ""

    # Этап 1 – первичные классы Σ1j
    primary_classes = build_primary_classes(states, inputs, outputs)
    text += "=== Этап 1: Первичное разбиение Σ1j ===\n"
    for j, block in enumerate(primary_classes, start=1):
        text += f"Σ1{j} = {sorted(block)}\n"

    # Этап 2 – уточнение классов Σ(i+1)j до стабилизации
    current = primary_classes
    step = 1

    while True:
        text += f"\n=== Шаг {step}: Σ{step}j ===\n"
        for j, block in enumerate(current, start=1):
            text += f"Σ{step}{j} = {sorted(block)}\n"

        # Строим следующее разбиение
        new = refine_partition(current, inputs, next_state)

        # Проверяем условие остановки: разбиение больше не меняется
        if normalize_partition(new) == normalize_partition(current):
            text += f"\nРазбиение стабилизировалось на шаге {step+1}\n"
            text += f"\n=== Шаг {step+1}: Σ{step+1}j ===\n"
            for j, block in enumerate(new, start=1):
                text += f"Σ{step+1}{j} = {sorted(block)}\n"
            final = new
            break

        current = new
        step += 1

    # Этап 3 – построение минимального автомата Ã
    text += "\n=== Этап 3: Минимальный автомат ===\n"

    P_equiv = final  # устойчивое разбиение Pэкв
    for j, block in enumerate(P_equiv, start=1):
        text += f"Σ{j} = {sorted(block)}\n"

    # Новые состояния S(j), каждый класс Σj -> одно состояние
    text += "\nНовые состояния:\n"
    new_states = {}
    for j, block in enumerate(P_equiv, start=1):
        # Представитель класса – минимальное по номеру состояние
        rep = sorted(block)[0]
        new_states[j] = rep
        text += f"S({j}) = {sorted(block)} (представитель {rep})\n"

    # Таблица минимального автомата:
    # для каждого нового состояния S(j) показываем:
    #   - строку выходов y по всем входам,
    #   - в какие новые состояния S(·) идут переходы по этим входам.
    text += "\nТаблица минимального автомата:\n\n"
    for j, rep in new_states.items():
        row_y = [outputs[rep][a] for a in inputs]
        row_s = [partition_to_state_map(P_equiv)[next_state[rep][a]] + 1 for a in inputs]
        text += f"S({j})  |  y = {row_y}   ->   {['S('+str(x)+')' for x in row_s]}\n"

    return text


# ------------------------- Streamlit UI -----------------------------

st.title("Минимизация автомата Мили (алгоритм из вашей методички)")

st.write("Введите таблицу автомата или загрузите дефолтный пример.")

col1, col2 = st.columns(2)

with col1:
    num_states = st.number_input("Количество состояний", min_value=1, max_value=20, value=9)
    num_inputs = st.number_input("Количество входов", min_value=1, max_value=10, value=3)
    max_output = st.number_input("Максимальный выходной сигнал", min_value=0, max_value=9, value=1)

with col2:
    if st.button("Загрузить дефолтный пример из тетради"):
        # Базовые словари для примера из тетради
        outputs_default = {
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
        next_state_default = {
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
        # Сохраняем пример сразу в словари, которыми пользуется алгоритм
        st.session_state["outputs"] = outputs_default
        st.session_state["next_state"] = next_state_default

        st.success("Дефолтный пример загружен!")

# ------------ Таблицы для ввода --------------

st.subheader("Введите таблицу выходов y(t):")

if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}

# Гарантируем наличие всех нужных ячеек в словаре outputs
outputs_dict = {}
for s in range(1, num_states + 1):
    row = st.session_state["outputs"].get(s, {})
    new_row = {}
    for a in range(1, num_inputs + 1):
        new_row[a] = row.get(a, 0)
    outputs_dict[s] = new_row
st.session_state["outputs"] = outputs_dict

# Шапка таблицы выходов
header_cols = st.columns(num_inputs + 1)
header_cols[0].markdown("**s(t)**")
for a_idx, a in enumerate(range(1, num_inputs + 1), start=1):
    header_cols[a_idx].markdown(f"**a{a}**")

# Строки таблицы выходов: используем number_input для каждой ячейки
for s in range(1, num_states + 1):
    row_cols = st.columns(num_inputs + 1)
    row_cols[0].markdown(f"**{s}**")
    for a_idx, a in enumerate(range(1, num_inputs + 1), start=1):
        cell_key = f"out_{s}_{a}"
        current_val = st.session_state["outputs"][s][a]
        # number_input с фиксированным key: состояние значения хранится в session_state,
        # поэтому оно НЕ сбрасывается при первом вводе
        new_val = row_cols[a_idx].number_input(
            label="",
            min_value=0,
            max_value=max_output,
            value=int(current_val),
            key=cell_key,
        )
        st.session_state["outputs"][s][a] = int(new_val)

st.subheader("Введите таблицу переходов s(t+1):")

if "next_state" not in st.session_state:
    st.session_state["next_state"] = {}

next_state_dict = {}
for s in range(1, num_states + 1):
    row = st.session_state["next_state"].get(s, {})
    new_row = {}
    for a in range(1, num_inputs + 1):
        new_row[a] = row.get(a, 1)
    next_state_dict[s] = new_row
st.session_state["next_state"] = next_state_dict

# Шапка таблицы переходов
header_cols_next = st.columns(num_inputs + 1)
header_cols_next[0].markdown("**s(t)**")
for a_idx, a in enumerate(range(1, num_inputs + 1), start=1):
    header_cols_next[a_idx].markdown(f"**a{a}**")

# Строки таблицы переходов
for s in range(1, num_states + 1):
    row_cols = st.columns(num_inputs + 1)
    row_cols[0].markdown(f"**{s}**")
    for a_idx, a in enumerate(range(1, num_inputs + 1), start=1):
        cell_key = f"next_{s}_{a}"
        current_val = st.session_state["next_state"][s][a]
        new_val = row_cols[a_idx].number_input(
            label="",
            min_value=1,
            max_value=num_states,
            value=int(current_val),
            key=cell_key,
        )
        st.session_state["next_state"][s][a] = int(new_val)

# ------------ RUN BUTTON -------------

if st.button("Запустить минимизацию"):
    st.write("### Результаты минимизации:")

    # Берём актуальные словари из session_state (они заполняются через number_input выше).
    outputs_dict = st.session_state["outputs"]
    next_state_dict = st.session_state["next_state"]

    # Проверка корректности таблицы переходов:
    # все целевые состояния должны быть в диапазоне [1, num_states]
    invalid_targets = []
    for s, row in next_state_dict.items():
        for a, t in row.items():
            if not (1 <= t <= num_states):
                invalid_targets.append((s, a, t))

    if invalid_targets:
        st.error(
            "В таблице переходов есть ссылки на несуществующие состояния. "
            "Проверьте, что все значения s(t+1) находятся в диапазоне от 1 до "
            f"{num_states} включительно."
        )
        # Покажем несколько первых ошибок для наглядности
        sample = invalid_targets[:5]
        st.write(
            "Примеры некорректных ячеек (строка = состояние s(t), столбец = вход a):"
        )
        st.table(
            pd.DataFrame(
                sample, columns=["s(t)", "вход a", "s(t+1) (некорректно)"]
            )
        )
        st.stop()

    result = run_minimization(
        inputs=list(range(1, num_inputs+1)),
        states=list(range(1, num_states+1)),
        outputs=outputs_dict,
        next_state=next_state_dict
    )

    st.text(result)
