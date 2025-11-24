# 關鍵字提取邏輯說明 (Keyword Extraction Logic)

本專案採用 **Rule-Based（規則式）** 結合 **Dependency Parsing（依存句法分析）** 的方法，旨在從餐廳評論中精準提取五大面向的評價關鍵字。

## 1. 核心邏輯：Aspect-Based Extraction (基於面向的提取)

我們關注以下五個面向，並為每個面向定義了種子關鍵字 (Seed Keywords)：

*   **Taste (口味)**: `food`, `flavor`, `delicious`, `tasty`, `chicken`, `pizza`...
*   **Price (價格)**: `price`, `cost`, `expensive`, `cheap`, `value`...
*   **Environment (環境)**: `atmosphere`, `vibe`, `clean`, `dirty`, `noisy`...
*   **Service (服務)**: `service`, `staff`, `waiter`, `friendly`, `rude`...
*   **Waiting Time (等待時間)**: `wait`, `time`, `queue`, `slow`, `fast`...

## 2. NLP 處理流程 (使用 Spacy)

當程式讀入一句評論時，會經過以下步驟：

### Step A: 詞形還原 (Lemmatization)
將單字還原成原型，以確保能與種子字典匹配。
*   例如：評論中的 "dishes" 會被還原為 "dish"，"prices" 還原為 "price"。

### Step B: 依存句法分析 (Dependency Parsing)
利用 Spacy 分析句子結構，找出與種子字相關的修飾語。主要採用以下兩種策略：

#### **策略一：名詞 + 形容詞修飾 (Adjective Modifier)**
*   **情境**：顧客直接用形容詞修飾名詞。
*   **例子**："I had a **tasty burger**."
*   **程式邏輯**：
    1.  找到種子字 "burger" (名詞)。
    2.  檢查其子節點 (children)。
    3.  若子節點為形容詞 (`ADJ`)，如 "tasty"，則提取 **"tasty burger"**。

#### **策略二：主詞 + 動詞 + 形容詞補語 (Subject + Verb + Adjective Complement)**
*   **情境**：名詞作為主詞，後接 Be 動詞與形容詞。
*   **例子**："The **service** was **slow**."
*   **程式邏輯**：
    1.  找到種子字 "service" (名詞)。
    2.  確認其為主詞 (`nsubj`)。
    3.  找到其父節點 (head) 動詞，如 "was"。
    4.  從動詞找尋形容詞補語 (`acomp`)，如 "slow"。
    5.  提取 **"service was slow"**。

*(註：原本的策略三「僅提取關鍵字」因雜訊過多已被移除)*

## 3. 情感判斷 (Sentiment Assignment)

提取出的關鍵字情感傾向直接與該則評論的 **星星數 (Star Rating)** 綁定：

*   **Positive (正向)**: 4 星 或 5 星
*   **Negative (負向)**: 1 星 或 2 星
*   **Neutral (中立)**: 3 星

**設計考量**：
雖然單句情感可能與整體評分不完全一致（例如 3 星評論中包含好壞參半的評價），但在 1 星或 5 星的極端評價中，這種強關聯性通常非常準確，適合用來快速建立初步的訓練資料集。


---

這是一個非常好的問題。關鍵字提取（Keyword Extraction）是這個專案的核心，我使用了 Rule-Based（規則式） 結合 Dependency Parsing（依存句法分析） 的方法。

簡單來說，我們不只是「找單字」，而是去理解「句子結構」，找出描述該面向的形容詞或片語。

以下是詳細的技術解說：

1. 核心邏輯：Aspect-Based Extraction (基於面向的提取)
我們關注五個面向：Taste (口味), Price (價格), Environment (環境), Service (服務), Waiting Time (等待時間)。

我首先定義了一個「種子字典 (Seed Dictionary)」，例如：

Taste: "food", "flavor", "delicious", "chicken", "pizza"...
Service: "staff", "waiter", "service", "rude", "friendly"...
2. NLP 處理流程 (使用 Spacy)
當程式讀入一句評論（例如："The chicken was incredibly juicy."）時，會經過以下步驟：

Step A: 詞形還原 (Lemmatization)
將單字還原成原型。

例如：評論寫 "dishes" 或 "prices"，程式會還原成 "dish" 和 "price"，這樣就能跟我們的種子字典對上。
Step B: 依存句法分析 (Dependency Parsing) - 這是最關鍵的一步
我們不只是看到 "chicken"，我們想知道 "chicken" 怎麼樣。Spacy 會分析字與字之間的關係。

我設計了兩套主要的提取策略（Strategy）：

策略一：名詞 + 形容詞修飾 (Adjective Modifier)

情境：顧客直接用形容詞修飾名詞。
例子："I had a tasty burger."
程式邏輯：
找到種子字 "burger" (Noun)。
檢查它的 "孩子" (children) 節點。
如果孩子是形容詞 (ADJ)，例如 "tasty"，就把它們組合成 "tasty burger"。
策略二：主詞 + 動詞 + 形容詞補語 (Subject + Verb + Adjective Complement)

情境：名詞當主詞，後面接 Be 動詞和形容詞。
例子："The service was slow."
程式邏輯：
找到種子字 "service" (Noun)。
發現它是句子的主詞 (nsubj)。
找到它的 "父親" (head) 動詞，這裡是 "was"。
再從動詞找它的 "形容詞補語" (acomp)，這裡是 "slow"。
組合成 "service was slow"。
3. 情感判斷 (Sentiment Assignment)
目前的邏輯是將提取出的關鍵字與該則評論的 星星數 (Star Rating) 綁定：

5星 / 4星 -> 標記為 Positive (正向)
1星 / 2星 -> 標記為 Negative (負向)
3星 -> 標記為 Neutral (中立)
為什麼這樣做？ 雖然 "Food was good but service was bad" 這種句子在 3 星評論中會導致混淆（例如 "service was bad" 被標記為 Neutral），但在大量的 1 星或 5 星評論中，這種強關聯性通常是準確的，適合用來做初步的訓練資料標註。

總結
這個方法的優點是精準度高且可解釋性強。我們只提取我們關心的面向（如價格、口味），並且透過語法分析抓出完整的「評價片語」（如 "overpriced food"），而不僅僅是零散的單字。