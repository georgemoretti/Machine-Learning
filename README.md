# Лабораторная работа: Исследование методов градиентного спуска для задачи регрессии

В рамках этой лабораторной работы мы исследуем различные методы градиентного спуска для решения задачи прогнозирования цен на автомобили. Работа включает полный цикл машинного обучения: от анализа данных до сравнения результатов различных алгоритмов.

## Основные этапы работы:

1. **Анализ и предобработка данных**:
   - Данные загружаются из файла `autos.csv` и исследуются с помощью визуализации (например, boxplot'ы для обнаружения выбросов).
   - Выбросы в целевой переменной (`price`) удаляются с использованием метода межквартильного размаха (IQR).
   - Создаются новые признаки, такие как логарифм мощности автомобиля (`log_powerPS`), квадратный корень возраста автомобиля (`age_sqrt`) и категориальные признаки через one-hot encoding.
   - Удаляются ненужные признаки, которые не влияют на качество модели или вызывают проблемы (например, `brand`, `model`).

2. **Создание новых признаков и масштабирование**:
   - Категориальные признаки кодируются с использованием target encoding и one-hot encoding.
   - Числовые признаки масштабируются с помощью `StandardScaler`.
   - Добавляется столбец смещения (`bias`) для улучшения сходимости моделей.

3. **Обучение моделей**:
   - Реализованы четыре метода градиентного спуска:
     - Полный градиентный спуск (`VanillaGradientDescent`),
     - Стохастический градиентный спуск (`StochasticDescent`),
     - Градиентный спуск с моментом (`MomentumDescent`),
     - Адаптивный метод Adam.
   - Для каждого метода исследуется влияние гиперпараметров, таких как скорость обучения (`λ`) и размер батча.
   - Используются различные функции потерь: MSE, Log-Cosh и Huber.

4. **Оценка качества моделей**:
   - Модели оцениваются по метрикам MSE и $R^2$ на обучающей и валидационной выборках.
   - Построены графики для сравнения производительности методов (ошибки, $R^2$, скорость сходимости).
   - Проанализировано влияние регуляризации на качество моделей.

5. **Результаты и выводы**:
   - Методы Momentum и Adam показали наилучшие результаты по скорости сходимости и качеству предсказаний.
   - Функция потерь Log-Cosh продемонстрировала устойчивость к выбросам, но на "чистых" данных её преимущество минимально.
   - Регуляризация не дала значительных улучшений, что говорит о том, что данные уже достаточно хорошо подготовлены.
