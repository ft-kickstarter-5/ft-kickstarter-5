# ft-kickstarter-5
Lambda Project Kickstarter

To start locally:

```
git clone <repo>
<activate venv>
flask run
```

To build in container: 
```
docker build -t flask-app .
docker container run -p 8083:5000 flask-app
```

To deploy, using Heroku cli:
```
heroku container:push web --app <app-name>
heroku container:release web --app <app-name>
```

## For users with M1 chip

To build in container with M1 chip: 

```
docker buildx build --platform linux/amd64 -t flask-app .
docker container run -p 8083:5000 flask-app
```

To deploy, using Heroku CLI and M1 chip:

```
docker tag flask-app registry.heroku.com/<app-name>/web
docker push registry.heroku.com/<app-name>/web
heroku container:release web --app <app-name>
```
