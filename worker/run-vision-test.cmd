@echo off
curl -s -m 170 -X POST http://localhost:7777/api/chat -H "Content-Type: application/json" --data-binary "@C:\Users\Admin\QuillanWorker\data\vision-test-body.json" -o C:\Users\Admin\QuillanWorker\data\vs-out.txt
echo DONE >> C:\Users\Admin\QuillanWorker\data\vs-out.txt
