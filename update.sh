#!/bin/bash
 
DATE="$(date +%Y-%m-%d\ %H:%M:%S)"
COMMIT_MESSAGE="Repo Update"\ ${DATE}

echo "Github Repo uploading..."
git add --all
git commit -m " ${COMMIT_MESSAGE}"
git push --force -u origin master
echo "Github Blog done..."