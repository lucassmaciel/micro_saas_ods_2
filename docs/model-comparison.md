# Comparativo de Modelos LLM

## Contexto

Durante o desenvolvimento do projeto, testamos quatro modelos para geração de respostas do assistente. O objetivo era encontrar um modelo que equilibrasse qualidade de resposta, disponibilidade em nuvem e limites de uso aceitáveis.

---

## Modelos Avaliados

| Critério | LLaMA 3 70B | LLaMA 3 8B | Qwen 2.5 3B | Qwen 2.5 1.5B |
|---|---|---|---|---|
| **Qualidade das respostas** | Alta | Boa | Fraca | Fraca |
| **Execução** | Nuvem (Groq) | Nuvem (Groq) | Local | Local |
| **Rate limit** | Estourava rápido | Adequado | N/A | N/A |
| **Escalabilidade** | Limitada | Boa | Sem escala | Sem escala |
| **Resultado dos testes** | Descartado | **Escolhido** | Descartado | Descartado |

---

## Análise por Modelo

### LLaMA 3 70B
- Gerava respostas de alta qualidade e com boa compreensão de contexto
- Rodava via nuvem, sem necessidade de infraestrutura local
- **Problema:** o rate limit era atingido muito rapidamente, inviabilizando o uso em produção com múltiplos usuários simultâneos

### LLaMA 3 8B — Modelo Escolhido
- Qualidade de resposta satisfatória para o caso de uso do projeto
- Roda em nuvem com rate limit significativamente mais generoso do que o 70B
- Boa relação entre qualidade e disponibilidade
- Viável para produção sem risco iminente de throttling

### Qwen 2.5 3B
- Qualidade de resposta abaixo do esperado para o domínio do projeto
- Exige execução local, o que impõe limitações de escalabilidade e infraestrutura
- Descartado pela combinação de baixa qualidade e ausência de escala em nuvem

### Qwen 2.5 1.5B
- Qualidade ainda inferior ao Qwen 3B
- Mesmas limitações de execução local
- Descartado pelos mesmos motivos, com desempenho ainda mais fraco

---

## Decisão Final

**Modelo escolhido: LLaMA 3 8B**

O LLaMA 3 8B foi selecionado por rodar em nuvem com um rate limit adequado ao volume esperado da aplicação, entregando qualidade de resposta aceitável para o contexto do projeto. O 70B, apesar da superioridade técnica, tornava-se inviável pelo consumo acelerado de cota. Os modelos Qwen, além de não gerarem resultados satisfatórios, exigiam infraestrutura local, o que conflita com os requisitos de escalabilidade e deploy em nuvem do projeto.
