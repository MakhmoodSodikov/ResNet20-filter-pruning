# ResNet20-filter-pruning

My repo with ResNet20 implementation and filter-level pruning experiments 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hO1x4xUbOZhnIJAkCCkT_2Ywmj8wNTcr?usp=sharing)
## Структрура проекта

Проект состоит из основных частей:

* модуля `model` с имплементацией исходников модели ResNet20, 
* `dataloader` c имплементацией загрузчика датасета для обучения CIFAR-10,
* `pruning` c имплементацией Filter-Level Pruning'a сети ResNet20,
* и скрипта для обучения `train.py`

В `utils.py` содержатся необходимые утилиты и функции-помощники для визуализации и анализа данных.

В скрипте `config.py` содержатся константы конфигурации, которые можно свободно варьировать для самостоятельного запуска.

Больше примеров и непосредственно эксперименты находятся в ноутбуке `experiments.ipynb`.

*Автор - Содиков Махмуд, МФТИ, 2021.*
