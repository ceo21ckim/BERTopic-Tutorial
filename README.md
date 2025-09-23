# BERTopic-Tutorial
BERTopic-Tutorial

### 1. clone git repository
```bash
git clone https://github.com/ceo21ckim/BERTopic-Tutorial.git

cd BERTopic-Tutorial
```


### build Dockerfile
```bash
docker build -t bertopic:org .

docker run -itd --name topic -p 8888:8888 -v c:\Users\user bertopic:org
```
