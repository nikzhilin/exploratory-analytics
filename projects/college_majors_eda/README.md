# College Majors EDA

Данный проект посвящён исследованию карьерных исходов выпускников американских колледжей в разрезе специальностей: зарплаты, занятость, уровень безработицы, гендерный состав и влияние уровня образования.

Цель проекта — последовательно зафиксировать состав исходных наборов данных, выполнить техническую очистку, сформировать производные признаки и проверить статистические гипотезы о связях между категорией специальности, уровнем образования, гендерным распределением и экономическими результатами выпускников.

## Состав проекта

Проект организован как линейная аналитическая цепочка. Каждый ноутбук закрывает отдельный этап подготовки или интерпретации данных, а результат предыдущего этапа используется на следующем.

### Ноутбуки
1. `01_overview.ipynb` - общий обзор трёх наборов данных, их структуры и первичных ограничений.
2. `02_technical_cleaning.ipynb` - проверка дублей, пропусков, типов данных и логических несоответствий.
3. `03_feature_engineering.ipynb` - построение производных признаков: full_time_rate, salary_range_rate, Grad_premium, Grad_share, diploma_impact_rate и других.
4. `04_derived_features_analysis.ipynb` - проверка распределений и аналитической полезности производных признаков.
5. `05_salaries_analysis.ipynb` - статистические гипотезы о распределении зарплат по категориям специальностей, STEM vs Non-STEM, связях безработицы и занятости.
6. `06_degree_levels_analysis.ipynb` - статистические гипотезы о влиянии уровня образования (graduate vs non-graduate) на зарплату, безработицу и занятость.
7. `07_gender_analysis.ipynb` - статистические гипотезы о связи гендерного состава специальности с зарплатой, занятостью и безработицей.
8. `08_summary.ipynb` - итоговая интерпретация результатов и ключевые выводы по всем направлениям анализа.

### Данные
- `data/raw/majors_all_ages.csv` - агрегированные показатели по специальностям для выпускников всех возрастов: численность, занятость, безработица, медианные доходы.
- `data/raw/majors_degree_levels.csv` - показатели по специальностям в разрезе graduate/non-graduate: занятость, безработица, медианные доходы, зарплатная премия.
- `data/raw/majors_recent_graduates.csv` - показатели по недавним выпускникам: гендерный состав, тип занятости, доходы, качество рабочих мест.
- `data/processed/majors_all_ages_clean.parquet` — очищенная версия majors_all_ages.
- `data/processed/majors_all_ages_analytics.parquet` — аналитическая таблица с производными признаками.
- `data/processed/majors_degree_levels_clean.parquet` — очищенная версия majors_degree_levels.
- `data/processed/majors_degree_levels_analytics.parquet` — аналитическая таблица с производными признаками.
- `data/processed/majors_recent_graduates_clean.parquet` — очищенная версия majors_recent_graduates.
- `data/processed/majors_recent_graduates_analytics.parquet` — аналитическая таблица с производными признаками.

## Ключевые выводы

Ранжирование подтверждённых гипотез по размеру практического эффекта:

| # | Находка | Тест | Эффект |
|---|---------|------|--------|
| 1 | Специальности с преобладанием мужчин зарабатывают значимо больше ($50k vs $33k) | Манн–Уитни | r = 0.89 |
| 2 | Аспирантура увеличивает зарплату ($75k vs $55k) | Вилкоксон | r = 0.85 |
| 3 | Категория специальности определяет зарплату | Краскел–Уоллис | η² = 0.66 |
| 4 | Категория специальности определяет долю женщин | Краскел–Уоллис | η² = 0.62 |
| 5 | Аспирантура снижает безработицу (3.95% vs 5.38%) | Вилкоксон | r = 0.63 |
| 6 | Доля женщин отрицательно коррелирует с зарплатой | Спирмен | ρ = −0.66 |
| 7 | Медианная зарплата и занятость полный день связаны | Спирмен | ρ = 0.76 |
| 8 | STEM зарабатывает больше Non-STEM | Манн–Уитни | большой |

Доходы и условия занятости выпускников определяются тремя взаимосвязанными факторами — категорией специальности, гендерным составом направления и уровнем образования. Аспирантура и STEM-специализация дают измеримое преимущество, тогда как гендерная сегрегация в специальностях объясняет значительную часть наблюдаемого разрыва в зарплатах.

## Структура каталогов

```text
college_majors_eda/
├── artifacts/
│   └── profiles/
│       ├── majors_all_ages.html
│       ├── majors_all_ages_analytics.html
│       ├── majors_degree_levels.html
│       ├── majors_degree_levels_analytics.html
│       ├── majors_recent_graduates.html
│       └── majors_recent_graduates_analytics.html
├── data/
│   ├── raw/
│   │   ├── majors_all_ages.csv
│   │   ├── majors_degree_levels.csv
│   │   └── majors_recent_graduates.csv
│   └── processed/
│       ├── majors_all_ages_clean.parquet
│       ├── majors_all_ages_analytics.parquet
│       ├── majors_degree_levels_clean.parquet
│       ├── majors_degree_levels_analytics.parquet
│       ├── majors_recent_graduates_clean.parquet
│       └── majors_recent_graduates_analytics.parquet
├── notebooks/
│   ├── 01_overview.ipynb
│   ├── 02_technical_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_derived_features_analysis.ipynb
│   ├── 05_salaries_analysis.ipynb
│   ├── 06_degree_levels_analysis.ipynb
│   ├── 07_gender_analysis.ipynb
│   └── 08_summary.ipynb
├── reports/
│   └── report.md
├── config.yaml
└── README.md
```
