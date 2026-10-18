#!/usr/bin/env python3
"""Delete the 2 probe workflows + their executions. Backup first."""
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
shutil.copy(r"C:\Users\Admin\.n8n\database.sqlite",
            r"C:\Users\Admin\.n8n\database.sqlite.bak-probedelete")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
ids = ["97389779-a0eb-4e70-82c9-2ab0d117534c",
       "be2de7ea-e691-48ee-a26f-d3683d6bdd39"]
for wid in ids:
    name = db.execute("SELECT name FROM workflow_entity WHERE id=?",
                      (wid,)).fetchone()
    print("deleting:", name[0] if name else "(missing)")
    db.execute("DELETE FROM execution_data WHERE executionId IN "
               "(SELECT id FROM execution_entity WHERE workflowId=?)", (wid,))
    db.execute("DELETE FROM execution_entity WHERE workflowId=?", (wid,))
    db.execute("DELETE FROM workflow_entity WHERE id=?", (wid,))
db.commit()
print("remaining:", db.execute(
    "SELECT name FROM workflow_entity").fetchall())
