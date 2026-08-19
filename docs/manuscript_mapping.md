# Manuscript-to-code mapping

| Manuscript component | Command or module |
|---|---|
| Tile colour-template extraction, composition grading, and semantic taxonomy | `msckg extract-tile-pixels`, `msckg classify-tiles`, `msckg.tile_pixels`, `msckg.tile_taxonomy` |
| Six-relation MSCKG construction | `msckg build-kg`, `msckg.kg` |
| DistMult and six alternative KGE models | `msckg train-kge`, `msckg.kge` |
| T, P, TC, and PC trajectory encoding | `msckg prepare-features`, `msckg.features` |
| Three-fold XGBoost parameter selection | `msckg select-parameters` |
| Main 13-combination experiment | `msckg main` |
| Seven embedding-model comparison | `msckg embedding-comparison` |
| XGBoost, Random Forest, and RBF-SVM comparison | `msckg classifier-comparison` |
| Seeds 0–4 and 95% intervals | `msckg repeated-splits` |
| POI frequency, POI TF-IDF, tile-category one-hot, and Word2Vec | `msckg semantic-baselines` |
| Spatial block CV, regional hold-out, tile-ID frequency, and spatial-grid one-hot | `msckg spatial-validation` |
| No T/TC, No TC, No P/PC, and No PC | `msckg component-ablation` |
| Spatial-expansion sensitivity | `msckg expand-training`, `msckg spatial-expansion` |
| Expert validation of tile-category language | `msckg expert-validation` |
| Mean pooling versus GRU | `msckg gru`, `msckg plot --kind gru-line` |
| POI hit rate, heterogeneous-neighbour proximity, and spatial concentration | `msckg domain-indicators` |
| Manuscript figures | `msckg plot` |
