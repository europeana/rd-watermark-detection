docker-compose up -d

docker-compose exec ml_deployment bash

chmod 0600 keys

s3fs europeana-thumbnails-production /code/data -o passwd_file=keys -o url=https://s3.eu.cloud-object-storage.appdomain.cloud 



find /code/data -type f | sort -R | awk 'NR <= 100'



find /code/data -type f | wc -l

rsync -n -i --recursive /code/data /dev/null | wc -l

find /code/data -type f -print0 | xargs -0 -n 1000 -P 10 wc -l | awk '{total += $1} END {print total}'

ls -l /code/data | awk '{if ($1 ~ /^-/) count++} END {print count}'


