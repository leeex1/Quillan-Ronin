#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
GIT HISTORY RE-ATTACHMENT & AUTHOR EMAIL RESTORATION
=============================================================================
1. Saves backup tag `archive/pre-history-stitch-main` pointing to current `main`.
2. Re-attaches the 2,148 clean historical commits (`d816da16f9b7c6f4e3dc4e415a70b09543648ff8`,
   which contains 0 LFS pointers > 2GB and 0 secret hits) directly underneath the
   235 post-Oct-18 commits (`af6cbb65..main`).
3. Rewrites `crashoverridex@quillan.local` -> `jdxleex@gmail.com` across the
   re-attached commits so GitHub attributes all 2,380+ commits and 14.8M+ lines
   to the user's GitHub account (`leeex1`).
4. Leaves working tree and index 100% untouched (`git diff main new_head` is empty).
=============================================================================
"""

import os
import subprocess
from typing import Dict

REPO_DIR = r"C:\02_QUILLAN"
HIST_BASE = "d816da16f9b7c6f4e3dc4e415a70b09543648ff8"
ORPHAN_ROOT = "af6cbb65e40949989e0efc5503af734679964735"


def git_bytes(*args: str, input_bytes: bytes = None) -> bytes:
    return subprocess.check_output(["git"] + list(args), cwd=REPO_DIR, input=input_bytes)


def git_str(*args: str) -> str:
    return git_bytes(*args).decode("utf-8", errors="replace").strip()


def main():
    cur_main = git_str("rev-parse", "main")
    print(f"Current main HEAD: {cur_main}")

    # Create backup tag
    subprocess.run(["git", "tag", "-f", "archive/pre-history-stitch-main", cur_main], cwd=REPO_DIR, check=True)
    print("Saved backup tag: archive/pre-history-stitch-main")

    # Get all commits in `main` in topological order from oldest to newest
    commits = git_str("rev-list", "--topo-order", "--reverse", "main").splitlines()
    print(f"Commits on current main to replay onto {HIST_BASE[:8]}: {len(commits)}")

    # Also check if any commits in `d816da16` have `crashoverridex@quillan.local`
    hist_commits = git_str("rev-list", "--topo-order", "--reverse", HIST_BASE).splitlines()
    print(f"Historical commits in {HIST_BASE[:8]}: {len(hist_commits)}")

    sha_map: Dict[str, str] = {}

    # First pass: rewrite any commits in `d816da16` that used `@quillan.local` so all 2,383 commits use `jdxleex@gmail.com`
    for c in hist_commits:
        raw = git_bytes("cat-file", "commit", c)
        header, _, body = raw.partition(b"\n\n")
        header_lines = header.split(b"\n")
        new_header_lines = []
        changed = False
        for ln in header_lines:
            if ln.startswith(b"parent "):
                old_p = ln[7:].decode("ascii").strip()
                new_p = sha_map.get(old_p, old_p)
                if new_p != old_p:
                    changed = True
                new_header_lines.append(f"parent {new_p}".encode("ascii"))
            elif ln.startswith(b"author ") or ln.startswith(b"committer "):
                if b"crashoverridex@quillan.local" in ln or b"toolkit@local" in ln:
                    ln = ln.replace(b"crashoverridex@quillan.local", b"jdxleex@gmail.com")
                    ln = ln.replace(b"toolkit@local", b"jdxleex@gmail.com")
                    changed = True
                new_header_lines.append(ln)
            else:
                new_header_lines.append(ln)

        if not changed:
            sha_map[c] = c
        else:
            new_raw = b"\n".join(new_header_lines) + b"\n\n" + body
            new_sha = git_bytes("hash-object", "-t", "commit", "-w", "--stdin", input_bytes=new_raw).decode("ascii").strip()
            sha_map[c] = new_sha

    new_hist_base = sha_map[HIST_BASE]
    print(f"Historical base mapped: {HIST_BASE[:8]} -> {new_hist_base[:8]}")

    # Second pass: replay the 235 commits of `main` on top of `new_hist_base`
    for c in commits:
        raw = git_bytes("cat-file", "commit", c)
        header, _, body = raw.partition(b"\n\n")
        header_lines = header.split(b"\n")
        new_header_lines = []
        has_parent = False

        for ln in header_lines:
            if ln.startswith(b"tree "):
                new_header_lines.append(ln)
                if c == ORPHAN_ROOT:
                    new_header_lines.append(f"parent {new_hist_base}".encode("ascii"))
                    has_parent = True
            elif ln.startswith(b"parent "):
                has_parent = True
                old_p = ln[7:].decode("ascii").strip()
                new_p = sha_map.get(old_p, old_p)
                new_header_lines.append(f"parent {new_p}".encode("ascii"))
            elif ln.startswith(b"author ") or ln.startswith(b"committer "):
                ln = ln.replace(b"crashoverridex@quillan.local", b"jdxleex@gmail.com")
                ln = ln.replace(b"toolkit@local", b"jdxleex@gmail.com")
                new_header_lines.append(ln)
            else:
                new_header_lines.append(ln)

        new_raw = b"\n".join(new_header_lines) + b"\n\n" + body
        new_sha = git_bytes("hash-object", "-t", "commit", "-w", "--stdin", input_bytes=new_raw).decode("ascii").strip()
        sha_map[c] = new_sha

    new_main_head = sha_map[cur_main]
    print(f"New stitched main HEAD: {new_main_head}")

    # Verify trees are 100% identical
    old_tree = git_str("rev-parse", f"{cur_main}^{{tree}}")
    new_tree = git_str("rev-parse", f"{new_main_head}^{{tree}}")
    assert old_tree == new_tree, f"Tree mismatch! {old_tree} != {new_tree}"
    print(f"Verified identical root tree SHA: {new_tree}")

    # Update `refs/heads/main` pointer without touching working tree or index
    subprocess.run(["git", "update-ref", "refs/heads/main", new_main_head, cur_main], cwd=REPO_DIR, check=True)
    total_count = git_str("rev-list", "--count", "main")
    print(f"SUCCESS: Updated refs/heads/main -> {new_main_head[:8]} (Total commits on main: {total_count})")
    print("Author breakdown on main:")
    print(git_str("shortlog", "-sne", "main"))


if __name__ == "__main__":
    main()

