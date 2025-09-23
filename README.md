# BERTopic-Tutorial
BERTopic-Tutorial

### 1. clone git repository
```bash
git clone https://github.com/ceo21ckim/BERTopic-Tutorial.git

cd BERTopic-Tutorial
```


### 2.build Dockerfile
```bash
docker build -t bertopic:org .

docker run -itd --name topic -p 8888:8888 -v c:\Users\user:/workspace bertopic:org
```

### 3. Jupyter Notebook

```bash
exec -it bertopic /bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
