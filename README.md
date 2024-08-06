Структура файловой системы репозитория:
- <b>debug/</b> - ноутбуки с отладкой кода из каталога "src".
- <b>experiments/</b> - каталог с реализациями проводимых экспериментов: дообучение/обучение моделей, подбор гиперпараметров модели и т.п. Каждый эксперимент в отдельной папке. Обязательно логирование экспериментальных процессов в отдельные каталоги "logs" (для каждого каталога с экспериментом свой каталог с логами): пул гиперпарметров + полученные метрики качества результатов эксперимента.
- <b>notebooks/</b> - переиспользуемые ноутбуки: связанные с предобработкой датасетов, формированием баз данных и т.п.
- <b>src/</b> - исходный код проекта
  - <b>Reader/</b> - реализации вариантов Reader-части RAG-системы.
  - <b>Retriever/</b> - реализации варинатов Retriever-части RAG-системы.
  - <b>Scorer/</b> - реализации вариантов алгоритмов по оценке сложности запросов и оценке неопределённости между запросом и документами (контекстами) в базе данных. 
  - <b>utils/</b> - вспомогательный инструментарий.
- <b>tests/</b> - модульные/интеграционные тесты для кода из каталога "src".
- <b>models/</b> - каталог с нейросетевыми моделями (в репозиторий не пушим)
- <b>data/</b> - каталог с датасетами/базми данных (в репозиторий не пушим)
- <b>docs/</b> - документация проекта.

  
Полезные материалы (структура ML-проекта): https://drive.google.com/file/d/1g0tzALqKygFTtzA-C5l5ZOdC9tKiUTzc/view?usp=sharing
