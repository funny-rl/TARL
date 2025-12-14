Action repetition

$$
\begin{align}
\tilde{Q}^\text{max}_k(s, a) &= \sum^{k-1}_{j=0} \gamma^{j}r_j + \gamma^k \max_{a' \in A}{Q_k(s', a')} \\
\rightarrow \\
\tilde{Q}^\text{eps}_k(s, a) &= \sum^{k-1}_{j=0} \gamma^{j}r_j + \gamma^k \left((1 - \alpha) \max_{a' \in A}{Q_k(s', a')}+ \alpha \frac{1}{|A|}\sum_{a'' \in A} Q_k(s', a'') \right) \\
\end{align}
$$


$\alpha$를 단순히 휴리스틱 값으로 활용할 경우, 이론적 근거가 매우 떨어진다. 따라서 $\alpha$를 특정 이론에 따라 조정할 수 있는 방법이 필요하다. 

기본 변수들은 위와 같이 정의한다. 여기서 $\beta$는 

$$
\max Q_k = \max_{a' \in A}{Q_k(s', a')} \\
\text{mean } Q_k = \frac{1}{|A|}\sum_{a'' \in A} Q_k(s', a'')\\
\min Q_k = \min_{a''' \in A}{Q_k(s', a''')} \\
$$

하나의 방법으로는 우리가 제안하는 target인 $(1 - \alpha) \max_{a' \in A}{Q_k(s', a')}+ \alpha \frac{1}{|A|}\sum_{a'' \in A} Q_k(s', a'')$가 기존 target인 $\max_{a' \in A}{Q_k(s', a')}$에 과하게 멀어지지 않는다는 보장이 필요하다. 따라서 Min-Max 정규화를 통한 scaling이 진행된 Q value에 대해 $\beta$의 값이 되도록 강제하는 방법이 있다. 

전제조건 (1, 2)을 통해 $\alpha$의 범위를 결정할 수 있다. 

$$
\begin{align}
1. \text{ mean } Q_k &< \max Q_k \\
2. \text{ mean } Q_k &\leq (1 - \alpha) \max Q_k + \alpha  \text{ mean } Q_k \leq  \max Q_k \\
0 &\leq (1 - \alpha) \max Q_k + (\alpha - 1) \text{ mean } Q_k \leq \max Q_k - \text{ mean } Q_k \\
(1 - \alpha) \text{ mean } Q_k  &\leq (1 - \alpha) \max Q_k \rightarrow (\alpha \leq 1)\\
(1 - \alpha) \max Q_k + (\alpha - 1) \text{ mean } Q_k &\leq  \max Q_k - \text{ mean } Q_k \\
\alpha \text{ mean } Q_k  &\leq \alpha \max Q_k \rightarrow (0 \leq \alpha)\\
\rightarrow (0 \leq \alpha \leq 1 )\\
\end{align}
$$

$$
\begin{align}
\beta &= \frac{\left((1 - \alpha) \max Q_k + \alpha  \text{ mean } Q_k \right) - \text{ mean } Q_k }{\max Q_k - \text{ mean } Q_k } \\ 
\beta &= \frac{(1 - \alpha) \max Q_k +  (\alpha - 1)\text{ mean } Q_k }{\max Q_k - \text{ mean } Q_k } \\ 
\beta &= \frac{(1 - \alpha) \max Q_k -  (1 - \alpha)\text{ mean } Q_k }{\max Q_k - \text{ mean } Q_k } \\ 
\beta &= \frac{(1 - \alpha)(\max Q_k - \text{ mean } Q_k) }{\max Q_k - \text{ mean } Q_k } \\ 
\beta &= 1 - \alpha \rightarrow (0 \leq \beta \leq 1 ) \\ 
\end{align}
$$ 

1. $1 \leq k \leq 10$ 인 각 repetition에 대해서 $\beta$를 동일하게 고정하는 경우, repetition network가 각 repetition에 대해 $\beta \max Q_k$를 추정하게되어 $Q_k$ 간의 대소 관계가 변하지 않음?  $\rightarrow$ 아님, 여기서 정의한 $\beta$는 단순히 $\max Q_k$에서 얼마나 멀어지는지 계산하는 것이 아니라 정규화된 거리를 기준으로 얼마나 멀어지는지 계산하기 때문에 (Min-Max 구간에서 어디에 위치하는지 강제하는 것) 각 $k$마다 기존 target인 $\max Q_k$에서 멀어지는 비율이 다름. 즉, $\beta \max Q_k$를 유도하고자 한 것이 아님. $\beta \max Q_k$에 대해 계산하기 위해선 $\frac{(1 - \alpha) \max Q_k + \alpha \text{ mean } Q_k }{\max Q_k} = \beta$로 정의해야 함. 

2. Min-Max에 $\min Q_k$를 적용하지 않은 이유: $(1 - \alpha) \max Q_k + \alpha  \text{ mean } Q_k$의 범위가 $\text{mean } Q_k$부터 $\max Q_k$까지이기 때문임. 


우리는 우리가 제안하는 $\tilde{Q}^\text{eps}_k(s, a)$가 기존 Q 
 