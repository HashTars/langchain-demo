FROM ubuntu:latest
LABEL authors="Tars"

ENTRYPOINT ["top", "-b"]