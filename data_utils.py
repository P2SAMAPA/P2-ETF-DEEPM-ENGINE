# Instead of downloading all 21 at once, run 3 batches of 7 with 6-hour gaps
name: Staggered Seed
on:
  workflow_dispatch:
jobs:
  batch1:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: python seed_batch.py --tickers AGG,GDX,GLD,HYG,LQD,MBB,PFF
  batch2:
    needs: batch1
    runs-on: ubuntu-latest
    steps:
      - run: sleep 21600  # 6 hours
      - uses: actions/checkout@v3
      - run: python seed_batch.py --tickers QQQ,SLV,SPY,TLT,VNQ,XLE,XLF
  batch3:
    needs: batch2
    runs-on: ubuntu-latest
    steps:
      - run: sleep 43200  # 12 hours
      - uses: actions/checkout@v3
      - run: python seed_batch.py --tickers XLI,XLK,XLP,XLU,XLV,XLY,XME
