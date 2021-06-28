## Тестовое задание на позицию Computer Vision Engineer в OSAI

Для того, чтобы проверить решение, необходимо выполнить команды:

sudo docker/build.sh

sudo docker/run.sho
или
sudo docker/run_volume.sh (Тут надо будет изменить <ABSOLUTE_PATH> на путь, где лежит папка с загруженными картинками)

После этого запустится контейнер, в котором все работает)

#### Загрузка изображений

Для начала загрузки изображений, необходимо выполнить:

python utils/download_data.py

Все битые ссылки будут проигнорированы, оставшиеся загрузятся в папку data/images. После этого пройдет проверка всех файлов на то, что это изображение и, если какие-то файлы окажутся чем-то иным, то они удалятся. 

Процесс загрузки досаточно долгий, поэтому для удобства выкладываю результат, чтобы его можно было подсунуть в папку: https://disk.yandex.ru/d/yWjEBAhodVlImg

Архив необходимо разархивировать и положить в data.
Тогда, запуская docker/run_volume.sh создастся вольюм с этими фотками.

#### Обучение сети

Для обучения сети используется фреймворк Catalyst. 
Все конфиги находятся в файле configs/config.yaml

Было обучено 2 сети: resnet18 и mobilenet_v3_small. Для выбора, какую именно сетку обучать, в конфиге необходимо поменять параметры модели, а именно поставить arch='resnet'

Процесс обучения тоже небыстрый, поэтому финальную модельку я сразу закинул в weights/mobilenet_v3_small.py


#### Метрики

Для запуска теста необходимо выполнить: 

python utils/inference.py

Нельзя сказать, что результаты обучения получились ошеломительные:

f1_macro=0.224
acc=0.244,

но если сравнить с результатами, которые были бы предсказаны просто взвешенным рандомом (вероятность_выбора_каждого_класса = (доля_класса_в_выборке)), то увидим, что рандом дал:

f1_macro~0.07
acc~0.07,
что говорит нам о том, что простейшая модель на не самых качественных данных дает предикт в 3-4 раза лучше, чем пальцем в небо. 



