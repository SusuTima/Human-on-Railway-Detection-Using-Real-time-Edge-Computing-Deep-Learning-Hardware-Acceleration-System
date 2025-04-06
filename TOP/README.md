除了下載TOP裡所有資料，亦須在 Vivado 呼叫 Block Memory Gnerator 的 IP。<br/>
BRAM_Control會呼叫到 12 個 BRAM_Big 及 32 個 BRAM_small (共44個)，設定如下(以下除了Write/Read Depth其他兩者設定皆一致)：<br/>
Basic: 
  - Interface Type: Native
  - Memory Type: True Dual Port RAM
<br/>Port Options: (for both Port A and Port B)
  - Write/Read Width: 16
  - Write/Read Depth
    --BRAM_Big: 6724
    --BRAM_small: 1450
    --disable "Primitives Output Register" and "Core Output Register"
<br/>
- **Interface Type:** Native  
- **Memory Type:** True Dual Port RAM  
- **Port Options (Port A and Port B):**
  - **Write/Read Width:** 16
  - **Write/Read Depth:**
    - BRAM_Big: 6724
    - BRAM_Small: 1450
- **Registers:**
  - Primitives Output Register: Disabled
  - Core Output Register: Disabled

詳請可參考BRAM_Settings.png
