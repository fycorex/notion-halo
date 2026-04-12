#!/bin/bash
cd ~/notion-halo
npm run dev:local
npm run dev:halo
git add --all
git commit -m "docs: update docs"
git push
