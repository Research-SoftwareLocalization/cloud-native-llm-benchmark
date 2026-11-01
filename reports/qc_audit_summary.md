# Cloud-Native Software Localization: Unified Quality Audit

**Audit Snapshot:** `2026-11-01 05:00 UTC`  
**Audited Suites (8):** `baseline_gemma_9b, baseline_gpt4o, baseline_llama_8b, baseline_mistral_nemo, baseline_qwen_7b, repaired_gpt4o, repaired_llama_8b, repaired_qwen_7b`  
**Corpus Size per Language:** `10,228` strings  
**Total Confirmed Defect Records:** `45517`  

### Empirical Failure Mode & Invariant Breakdown

| Model Suite | Lang | Lines | Script Contam. | Broken %1 | Broken & | Corrupt () | Chatter | Copy Fallback | Total Flawed | Defect Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_gemma_9b` | `ar` | 10228 | 55 | 7 | 1 | 0 | 0 | 314 | **377** | **3.69%** |
| `baseline_gemma_9b` | `bn` | 10228 | 51 | 3 | 1 | 0 | 0 | 575 | **630** | **6.16%** |
| `baseline_gemma_9b` | `de` | 10228 | 2 | 5 | 0 | 0 | 1 | 183 | **191** | **1.87%** |
| `baseline_gemma_9b` | `es` | 10228 | 0 | 4 | 1 | 0 | 0 | 113 | **118** | **1.15%** |
| `baseline_gemma_9b` | `fr` | 10228 | 0 | 3 | 1 | 0 | 0 | 99 | **103** | **1.01%** |
| `baseline_gemma_9b` | `hi` | 10228 | 6 | 2 | 1 | 0 | 0 | 335 | **344** | **3.36%** |
| `baseline_gemma_9b` | `ja` | 10228 | 8 | 2 | 1 | 0 | 0 | 277 | **288** | **2.82%** |
| `baseline_gemma_9b` | `mr` | 10228 | 78 | 3 | 0 | 0 | 0 | 1883 | **1964** | **19.20%** |
| `baseline_gemma_9b` | `pa` | 10228 | 473 | 6 | 0 | 0 | 0 | 7487 | **7965** | **77.87%** |
| `baseline_gemma_9b` | `ta` | 10228 | 171 | 2 | 1 | 0 | 0 | 1789 | **1963** | **19.19%** |
| `baseline_gemma_9b` | `te` | 10228 | 373 | 1 | 0 | 0 | 0 | 7371 | **7745** | **75.72%** |
| `baseline_gemma_9b` | `zh` | 10228 | 9 | 5 | 1 | 0 | 0 | 185 | **200** | **1.96%** |
| `baseline_gpt4o` | `ar` | 10228 | 0 | 1 | 0 | 0 | 0 | 51 | **52** | **0.51%** |
| `baseline_gpt4o` | `bn` | 10228 | 6 | 0 | 0 | 0 | 0 | 54 | **60** | **0.59%** |
| `baseline_gpt4o` | `de` | 10228 | 0 | 0 | 0 | 0 | 1 | 208 | **209** | **2.04%** |
| `baseline_gpt4o` | `es` | 10228 | 0 | 0 | 0 | 0 | 0 | 161 | **161** | **1.57%** |
| `baseline_gpt4o` | `fr` | 10228 | 0 | 1 | 0 | 0 | 0 | 198 | **199** | **1.95%** |
| `baseline_gpt4o` | `hi` | 10228 | 4 | 0 | 0 | 0 | 0 | 67 | **71** | **0.69%** |
| `baseline_gpt4o` | `ja` | 10228 | 1 | 0 | 0 | 0 | 0 | 53 | **54** | **0.53%** |
| `baseline_gpt4o` | `mr` | 10228 | 1 | 1 | 0 | 0 | 0 | 77 | **79** | **0.77%** |
| `baseline_gpt4o` | `pa` | 10228 | 38 | 0 | 0 | 0 | 0 | 56 | **94** | **0.92%** |
| `baseline_gpt4o` | `ta` | 10228 | 26 | 1 | 0 | 0 | 0 | 94 | **120** | **1.17%** |
| `baseline_gpt4o` | `te` | 10228 | 5 | 1 | 0 | 0 | 0 | 33 | **39** | **0.38%** |
| `baseline_gpt4o` | `zh` | 10228 | 0 | 0 | 0 | 0 | 0 | 31 | **31** | **0.30%** |
| `baseline_llama_8b` | `ar` | 10228 | 0 | 163 | 1 | 0 | 0 | 88 | **252** | **2.46%** |
| `baseline_llama_8b` | `bn` | 10228 | 0 | 111 | 0 | 5 | 0 | 334 | **450** | **4.40%** |
| `baseline_llama_8b` | `de` | 10228 | 0 | 107 | 0 | 0 | 1 | 103 | **210** | **2.05%** |
| `baseline_llama_8b` | `es` | 10228 | 0 | 100 | 0 | 0 | 0 | 75 | **175** | **1.71%** |
| `baseline_llama_8b` | `fr` | 10228 | 0 | 78 | 1 | 0 | 0 | 94 | **173** | **1.69%** |
| `baseline_llama_8b` | `hi` | 10228 | 0 | 121 | 1 | 0 | 0 | 60 | **182** | **1.78%** |
| `baseline_llama_8b` | `ja` | 10228 | 0 | 267 | 1 | 0 | 0 | 49 | **317** | **3.10%** |
| `baseline_llama_8b` | `mr` | 10228 | 0 | 120 | 0 | 2 | 0 | 73 | **195** | **1.91%** |
| `baseline_llama_8b` | `pa` | 10228 | 0 | 144 | 0 | 2 | 0 | 312 | **458** | **4.48%** |
| `baseline_llama_8b` | `ta` | 10228 | 0 | 76 | 1 | 1 | 0 | 52 | **130** | **1.27%** |
| `baseline_llama_8b` | `te` | 10228 | 0 | 132 | 1 | 0 | 0 | 21 | **154** | **1.51%** |
| `baseline_llama_8b` | `zh` | 10228 | 0 | 273 | 1 | 0 | 0 | 38 | **312** | **3.05%** |
| `baseline_mistral_nemo` | `ar` | 10228 | 0 | 1218 | 0 | 0 | 0 | 34 | **1252** | **12.24%** |
| `baseline_mistral_nemo` | `bn` | 10228 | 0 | 632 | 1 | 0 | 0 | 86 | **719** | **7.03%** |
| `baseline_mistral_nemo` | `de` | 10228 | 0 | 1125 | 1 | 0 | 0 | 47 | **1173** | **11.47%** |
| `baseline_mistral_nemo` | `es` | 10228 | 0 | 1126 | 1 | 0 | 0 | 43 | **1170** | **11.44%** |
| `baseline_mistral_nemo` | `fr` | 10228 | 0 | 1216 | 1 | 0 | 1 | 55 | **1272** | **12.44%** |
| `baseline_mistral_nemo` | `hi` | 10228 | 0 | 511 | 0 | 0 | 0 | 68 | **579** | **5.66%** |
| `baseline_mistral_nemo` | `ja` | 10228 | 0 | 687 | 1 | 0 | 0 | 17 | **705** | **6.89%** |
| `baseline_mistral_nemo` | `mr` | 10228 | 0 | 827 | 0 | 0 | 0 | 60 | **887** | **8.67%** |
| `baseline_mistral_nemo` | `pa` | 10228 | 0 | 576 | 0 | 8 | 0 | 179 | **763** | **7.46%** |
| `baseline_mistral_nemo` | `ta` | 10228 | 0 | 657 | 0 | 0 | 0 | 30 | **687** | **6.72%** |
| `baseline_mistral_nemo` | `te` | 10228 | 0 | 783 | 0 | 2 | 1 | 89 | **875** | **8.55%** |
| `baseline_mistral_nemo` | `zh` | 10228 | 0 | 790 | 1 | 0 | 0 | 10 | **801** | **7.83%** |
| `baseline_qwen_7b` | `ar` | 10228 | 0 | 30 | 0 | 80 | 0 | 460 | **567** | **5.54%** |
| `baseline_qwen_7b` | `bn` | 10228 | 304 | 23 | 0 | 146 | 0 | 92 | **550** | **5.38%** |
| `baseline_qwen_7b` | `de` | 10228 | 24 | 15 | 0 | 0 | 1 | 123 | **163** | **1.59%** |
| `baseline_qwen_7b` | `es` | 10228 | 0 | 33 | 0 | 0 | 0 | 127 | **160** | **1.56%** |
| `baseline_qwen_7b` | `fr` | 10228 | 0 | 48 | 0 | 0 | 0 | 166 | **214** | **2.09%** |
| `baseline_qwen_7b` | `hi` | 10228 | 0 | 81 | 0 | 202 | 0 | 294 | **577** | **5.64%** |
| `baseline_qwen_7b` | `ja` | 10228 | 0 | 161 | 0 | 0 | 0 | 148 | **309** | **3.02%** |
| `baseline_qwen_7b` | `mr` | 10228 | 0 | 67 | 0 | 99 | 0 | 169 | **335** | **3.28%** |
| `baseline_qwen_7b` | `pa` | 10228 | 0 | 45 | 0 | 222 | 0 | 1251 | **1515** | **14.81%** |
| `baseline_qwen_7b` | `ta` | 10228 | 0 | 48 | 0 | 205 | 0 | 638 | **890** | **8.70%** |
| `baseline_qwen_7b` | `te` | 10228 | 0 | 77 | 0 | 389 | 0 | 431 | **889** | **8.69%** |
| `baseline_qwen_7b` | `zh` | 10228 | 0 | 63 | 0 | 0 | 0 | 34 | **97** | **0.95%** |
| `repaired_gpt4o` | `ar` | 10228 | 0 | 1 | 0 | 0 | 0 | 0 | **1** | **0.01%** |
| `repaired_gpt4o` | `bn` | 10228 | 3 | 0 | 0 | 0 | 0 | 1 | **4** | **0.04%** |
| `repaired_gpt4o` | `de` | 10228 | 0 | 0 | 0 | 0 | 1 | 80 | **81** | **0.79%** |
| `repaired_gpt4o` | `es` | 10228 | 0 | 0 | 0 | 0 | 0 | 49 | **49** | **0.48%** |
| `repaired_gpt4o` | `fr` | 10228 | 0 | 0 | 0 | 0 | 0 | 84 | **84** | **0.82%** |
| `repaired_gpt4o` | `hi` | 10228 | 0 | 0 | 0 | 0 | 0 | 3 | **3** | **0.03%** |
| `repaired_gpt4o` | `ja` | 10228 | 1 | 0 | 0 | 0 | 0 | 4 | **5** | **0.05%** |
| `repaired_gpt4o` | `mr` | 10228 | 0 | 1 | 0 | 0 | 0 | 4 | **5** | **0.05%** |
| `repaired_gpt4o` | `pa` | 10228 | 13 | 0 | 0 | 0 | 0 | 2 | **15** | **0.15%** |
| `repaired_gpt4o` | `ta` | 10228 | 12 | 1 | 0 | 0 | 0 | 2 | **14** | **0.14%** |
| `repaired_gpt4o` | `te` | 10228 | 2 | 1 | 0 | 0 | 0 | 2 | **5** | **0.05%** |
| `repaired_gpt4o` | `zh` | 10228 | 0 | 0 | 0 | 0 | 0 | 1 | **1** | **0.01%** |
| `repaired_llama_8b` | `ar` | 10228 | 0 | 7 | 1 | 0 | 0 | 2 | **10** | **0.10%** |
| `repaired_llama_8b` | `bn` | 10228 | 0 | 4 | 0 | 0 | 0 | 5 | **9** | **0.09%** |
| `repaired_llama_8b` | `de` | 10228 | 0 | 4 | 0 | 0 | 0 | 27 | **31** | **0.30%** |
| `repaired_llama_8b` | `es` | 10228 | 0 | 5 | 0 | 0 | 0 | 30 | **35** | **0.34%** |
| `repaired_llama_8b` | `fr` | 10228 | 0 | 5 | 1 | 0 | 0 | 19 | **25** | **0.24%** |
| `repaired_llama_8b` | `hi` | 10228 | 0 | 5 | 1 | 0 | 0 | 3 | **9** | **0.09%** |
| `repaired_llama_8b` | `ja` | 10228 | 0 | 4 | 1 | 0 | 0 | 1 | **6** | **0.06%** |
| `repaired_llama_8b` | `mr` | 10228 | 0 | 3 | 0 | 1 | 0 | 0 | **4** | **0.04%** |
| `repaired_llama_8b` | `pa` | 10228 | 0 | 4 | 0 | 0 | 0 | 2 | **6** | **0.06%** |
| `repaired_llama_8b` | `ta` | 10228 | 0 | 4 | 1 | 0 | 0 | 0 | **5** | **0.05%** |
| `repaired_llama_8b` | `te` | 10228 | 0 | 3 | 1 | 0 | 0 | 0 | **4** | **0.04%** |
| `repaired_llama_8b` | `zh` | 10228 | 0 | 4 | 1 | 0 | 0 | 0 | **5** | **0.05%** |
| `repaired_qwen_7b` | `ar` | 10228 | 0 | 5 | 0 | 60 | 0 | 216 | **278** | **2.72%** |
| `repaired_qwen_7b` | `bn` | 10228 | 134 | 3 | 0 | 58 | 0 | 28 | **218** | **2.13%** |
| `repaired_qwen_7b` | `de` | 10228 | 2 | 1 | 0 | 0 | 0 | 44 | **47** | **0.46%** |
| `repaired_qwen_7b` | `es` | 10228 | 0 | 5 | 0 | 0 | 0 | 46 | **51** | **0.50%** |
| `repaired_qwen_7b` | `fr` | 10228 | 0 | 5 | 0 | 0 | 0 | 56 | **61** | **0.60%** |
| `repaired_qwen_7b` | `hi` | 10228 | 0 | 5 | 0 | 89 | 0 | 111 | **205** | **2.00%** |
| `repaired_qwen_7b` | `ja` | 10228 | 0 | 3 | 0 | 0 | 0 | 5 | **8** | **0.08%** |
| `repaired_qwen_7b` | `mr` | 10228 | 0 | 1 | 0 | 26 | 0 | 60 | **87** | **0.85%** |
| `repaired_qwen_7b` | `pa` | 10228 | 0 | 6 | 0 | 56 | 0 | 314 | **376** | **3.68%** |
| `repaired_qwen_7b` | `ta` | 10228 | 0 | 11 | 0 | 79 | 0 | 209 | **299** | **2.92%** |
| `repaired_qwen_7b` | `te` | 10228 | 0 | 4 | 0 | 173 | 0 | 77 | **253** | **2.47%** |
| `repaired_qwen_7b` | `zh` | 10228 | 0 | 3 | 0 | 0 | 0 | 1 | **4** | **0.04%** |

### Placeholder Failure Decomposition: Omission vs. Instruction-Induced Hallucination

| Model Suite | Total `%1` Errors | Dropped `%1` (Omission) | Injected `%1` (Prompt Hallucination) | Partial/Altered `%1` |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_gemma_9b` | **43** | 38 | **0** | 5 |
| `baseline_gpt4o` | **5** | 0 | **1** | 4 |
| `baseline_llama_8b` | **1692** | 90 | **1529** | 73 |
| `baseline_mistral_nemo` | **10148** | 221 | **9820** | 107 |
| `baseline_qwen_7b` | **691** | 60 | **582** | 49 |
| `repaired_gpt4o` | **4** | 0 | **0** | 4 |
| `repaired_llama_8b` | **52** | 20 | **12** | 20 |
| `repaired_qwen_7b` | **52** | 7 | **31** | 14 |
