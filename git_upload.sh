#!/bin/bash

# 获取当前时间，格式为 "YYYY/MM/DD HH:MM"
current_time=$(date +"%Y/%m/%d %H:%M")

# 如果没有提供 commit message 参数，则使用默认的 "fix bugs" 加上当前时间
if [ -z "$1" ]; then
  commit_message="fix bugs $current_time"
else
  commit_message="$1"
fi

# 执行 git 命令
git add .
git commit -m "$commit_message"
git push -u origin master
