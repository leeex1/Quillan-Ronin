@echo off
curl -s -m 235 -X POST http://localhost:7777/api/chat -H "Content-Type: application/json" --data-binary "{\"message\":\"You now carry your complete distilled kernel. Confirm from it: version, council size, swarm scale. Two sentences max.\",\"mode\":\"fulldive\",\"history\":[]}" -o C:\Users\Admin\QuillanWorker\data\dive-out.txt
echo DONE >> C:\Users\Admin\QuillanWorker\data\dive-out.txt
