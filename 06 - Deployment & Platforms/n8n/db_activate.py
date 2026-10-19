#!/usr/bin/env python3
"""Activate MAIN + report flags. (DB backed up by caller.)"""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r"C:\Users\Admin\.n8n\database.sqlite")
db.execute("UPDATE workflow_entity SET active=1 WHERE id='EmfoGhyIzSrktmIe'")
db.commit()
for name, active in db.execute("SELECT name, active FROM workflow_entity"):
    print(("ACTIVE " if active else "off    ") + name)
