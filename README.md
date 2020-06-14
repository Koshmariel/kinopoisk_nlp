Эта нейросеть была сделана в качестве теста по обработке натурального языка по мотивам известной базы обзоров фильмов IMDB. В то же время классификация обзоров по категориям положительный/отрицательный показалась мне слишком скучной. Поэтому нейросеть пытается оценивать обзоры по 10-бальной шкале.
В качестве базы данных для тренировки и теста взята база обзоров с сайта kinopoisk.ru Хотя Kinoposk так же как IMDB классифицирует обзоры только по категориям отрицательный/нейтральный/положительный многие авторы обзоров выставляют оценку фильму по 10-бальной шкале. Для загрузки текстов обзоров написан вэб-скрэппер scrapper.py, сохраняющие все обзоры длиннее 150 символов, в тексте которых присутствует оценка вида «n из 10». Для обхода капчи скрэппер использует бесплатыне проки-серверы, автоматически загружая их списки.
Перед тренировкой набор данных нормализуется в связи с большим количеством обзоров с высокими оценками во избежание смещения предсканий в их сторону.
Текст обзора очищается от не имеющих значения слов. Затем слова заменяются на их основы Snowball стеммером, показывающим лучший результат чем Porter стеммер.
Размер словаря ограничен 3.000 наиболее часто встречающимися в обзорах словами.
При обучении используется уменьшение коэфициента обучения если уменьшениие loss не происходит в течении 2 эпох.
Результаты, с моей точки зрения, получились не плохими. При намного меньшем объёме тренировочных данных (10.000 против 50.000 у IMDB) в своих оценках нейросеть ошибается в среднем на 1 балл.

This artificial neural network was inspired by the famous IMDB movie review dataset. I thought that classification into positive/negative categories is to boring that's why ANN is rating reviews on 10 point scale.
As training and testing dataset I took movie review base from kinopoisk.ru. While kinopoisk as IMDB classifies reviews only by positive/neutral/negative categories many authors include 10 point scale rating in their reviews. To download the dataset I made web-scrapper scrapper.py which takes reviews consisting of 150 or more character including rate in the form of  “n out of 10”. To avoid capcha scrapper uses free proxies by automatically downloading their lists.
Before training the dataset should be normalized because most reviews are high-rated to avoid bias towards it.
Review texts are cleaned form irrelevant words. Then the words are reduced to their stem by Snowball stemmer which shows better results than Porter stemmer.
Vocabulary size is limited to 3000 words most frequently used in reviews.
During training learning rate reduction callback is implemented if loss is not reduced for more than 2 epochs.
Results are not bad. While using much smaller training dataset (10k against IMDB 50k) the average error is around 1 point.
