## To build and run the docker container
```bash
docker build -t -f docker/dockerfile envisionhg_detector_docker .
```
Model Options: 'cnn_b' or 'lightgbm'
```bash
docker run --rm  -v path/to/input:/input -v path/to/output:/output envisionhg_detector_docker --input /input --output /output --model 'lightgbm'
```

## To upload the docker container
```bash
docker tag envisionhg_detector_docker USERNAME/envisionhgdetector_docker:latest
docker push USERNAME/envisionhgdetector_docker:latest
```

Then use it as (model = lightgbm or cnn_b)
```bash
docker pull USERNAME/envisionhgdetector_docker:latest
docker run --rm -v path/to/input/folder:/input  -v path/to/output/folder:/output USERNAME:latest --input /input --output /output  --model 'lightgbm'
```
