# Практическое задание на тему "Классификация комментариев на ревью"

Выполнил: Литвинов Михаил Юрьевич, группа МСТПР241.

## Подготовка окружения

Как и было предложено, работа с проектом происходит посредством развертывания отдельного venv-окружения:

```shell
python3 -m venv venv
```

Датасет ToxiCR подгружается с помощью Git-сабмодуля:

```shell
git submodule update --init
```

Установка всех зависимостей:

```
venv/Scripts/python -m pip install -r requirements.tvt
venv/Scripts/python -m pip install -r requirements_dev.tvr
```

## Работа с классическими методами ML

В первую очередь хотелось рассмотреть, как себя будут показывать простейшие алгоритмы с гиперпараметрами по умолчанию.
На тот момент никакой хитрой обработки входных данных не было: лишь отсечения пустых значений (`None`) и удаление дубликатов.

Основной интерес представляла логистическая регрессия из `sklearn` на основе tf-idf или bag-of-words - сначала был рассмотрен tf-idf.
По умолчанию логистическая регрессия из `sklearn` использует солвер `lbfgs` с ridge-регуляризацией (l2) и обратным коэффициентом 1.
Для такой конфигурации были получены следующие результаты:

```
Mean cross-entropy loss: 4.27361978342756
True non-toxic comments recognized: 10246
Non-toxic comments treated as toxic: 136
Toxic comments treated as non-toxic: 1394
True toxic comments recognized: 1128
Final F1-score: 0.595879556259905
```

Далее я вспомнил, что lasso-регуляризация (l1) может быть более эффективной в случае, когда у нас имеется большое количество признаков - для tf-idf и bag-of-words это вполне ожидаемо.

Заменив l2 на l1 и солвер `lbfgs` на `liblinear` (первый не работает с l1) я получил следующие результаты:

```
Mean cross-entropy loss: 3.1200217634565917
True non-toxic comments recognized: 10192
Non-toxic comments treated as toxic: 190
Toxic comments treated as non-toxic: 927
True toxic comments recognized: 1595
Final F1-score: 0.7406547480845136
```

Я пришел к выводу, что теперь стоило бы попробовать другую форму извлечения признаков - обычный bag-of-words вместо tf-idf.
Идействительно - показатели F1-метрики увеличились:

```
Для kfold=10, count, l1, liblinear:
Mean cross-entropy loss: 2.938462753049538
True non-toxic comments recognized: 10051
Non-toxic comments treated as toxic: 331
Toxic comments treated as non-toxic: 721
True toxic comments recognized: 1801
Final F1-score: 0.7739578856897292
```

Немного модифицировав обратный коэффициент регуляризации - заменив 1 на 1.35 - я еще больше увеличил F1-метрику:

```
Mean cross-entropy loss: 2.9049441665128515
True non-toxic comments recognized: 10022
Non-toxic comments treated as toxic: 360
Toxic comments treated as non-toxic: 680
True toxic comments recognized: 1842
Final F1-score: 0.7798475867908552
```

## Работа с трансформером CodeBERT

