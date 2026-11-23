# Ön deney özeti

Doğruluk (%), FP16'ya göre fark (yüzde puan) ve McNemar p değeri.
Çoklu karşılaştırma için Bonferroni eşiği: 0.05 / 32 = 0.0016

| olgu                            |   fp16 |   int8 |   nf4 |   int8 fark |   int8 p |   nf4 fark | nf4 p   |
|:--------------------------------|-------:|-------:|------:|------------:|---------:|-----------:|:--------|
| GENEL                           |   78.9 |   78.5 |  76.7 |        -0.4 |    0.222 |       -2.2 | <0.001  |
| anaphor_agreement               |   78.3 |   77.7 |  76.3 |        -0.7 |    0.824 |       -2   | 0.471   |
| argument_structure_ditransitive |   71   |   69.7 |  67   |        -1.3 |    0.454 |       -4   | 0.111   |
| argument_structure_transitive   |   62.7 |   63.3 |  57.7 |         0.7 |    0.832 |       -5   | 0.063   |
| binding                         |   72   |   70.3 |  70   |        -1.7 |    0.332 |       -2   | 0.451   |
| determiners                     |   75   |   73.3 |  76.7 |        -1.7 |    0.5   |        1.7 | 0.576   |
| ellipsis                        |   17   |   19.3 |  16   |         2.3 |    0.21  |       -1   | 0.678   |
| irregular_forms                 |   97.7 |   95.7 |  96.7 |        -2   |    0.07  |       -1   | 0.607   |
| island_effects                  |   86.3 |   87.3 |  82.3 |         1   |    0.607 |       -4   | 0.081   |
| nominalization                  |   83.7 |   82   |  81.3 |        -1.7 |    0.405 |       -2.3 | 0.281   |
| npi_licensing                   |   94.3 |   94.3 |  93.3 |         0   |    1     |       -1   | 0.549   |
| passives                        |   99.3 |   99.7 |  99.3 |         0.3 |    1     |        0   | 1.000   |
| quantifiers                     |  100   |  100   | 100   |         0   |    1     |        0   | 1.000   |
| relative_clauses                |   63.3 |   64.7 |  60.3 |         1.3 |    0.597 |       -3   | 0.281   |
| scrambling                      |   97.3 |   97.3 |  98.3 |         0   |    1     |        1   | 0.453   |
| subject_verb_agreement          |   74.3 |   73.3 |  69.7 |        -1   |    0.607 |       -4.7 | 0.029   |
| suspended_affixation            |   90.7 |   88.3 |  82.3 |        -2.3 |    0.039 |       -8.3 | <0.001  |

## Kaynak kullanımı

| varyant   |   dogruluk |   model_bellek_gb |   gecikme_ms_token |   tepe_bellek_gb |   yukleme_puanlama_sn |
|:----------|-----------:|------------------:|-------------------:|-----------------:|----------------------:|
| fp16      |      0.789 |             2.875 |             42.743 |            4.507 |                63.06  |
| int8      |      0.785 |             1.655 |            169.422 |            3.288 |               144.696 |
| nf4       |      0.767 |             1.045 |             60.035 |            2.761 |                85.257 |
