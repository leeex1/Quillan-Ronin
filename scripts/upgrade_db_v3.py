import sqlite3
DB = r'C:\02_QUILLAN\quillan.db'
con = sqlite3.connect(DB)
c = con.cursor()
c.execute("CREATE TABLE IF NOT EXISTS contradictions(id INTEGER PRIMARY KEY, agent TEXT, flagged_at TEXT, claim_a TEXT, claim_b TEXT, status TEXT)")
for ddl in ("ALTER TABLE decisions ADD COLUMN verified TEXT DEFAULT 'verified'",
            "ALTER TABLE memory_notes ADD COLUMN verified TEXT DEFAULT 'verified'"):
    try:
        c.execute(ddl)
    except Exception as e:
        print("skip:", e)
con.commit()
print("cols decisions:", [r[1] for r in c.execute("PRAGMA table_info(decisions)")])
print("cols notes:", [r[1] for r in c.execute("PRAGMA table_info(memory_notes)")])
con.close()
