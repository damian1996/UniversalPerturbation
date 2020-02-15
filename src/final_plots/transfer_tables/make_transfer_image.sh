convert $1_0_001_trained.png $1_0_005_trained.png +append top.png
convert $1_0_008_trained.png $1_0_01_trained.png +append mid.png
convert $1_0_05_trained.png $1_0_1_trained.png +append down.png
convert top.png mid.png down.png -append tranfer_tables_$1.png