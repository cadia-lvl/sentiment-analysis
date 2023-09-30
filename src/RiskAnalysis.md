# Risk Analysis for Sentiment-Analysis in Icelandic using Machine Translation

| Risk                                  | Likelihood (1-5)  | Impact (1-5)  | Responsibility    | Mitigation Strategy |
|---------------------------------------|-------------------|---------------|-------------------|---------------------|
| Resource Constraints (Time and Computing Power) | 4       | 5             | Eysteinn          | Prioritize key features and models that are critical to the project. Consider using cloud computing resources. |
| Training stops/Computer crashes       | 4                 | 4             | Ólafur            | Regular backups and distributed training could mitigate this risk. |
| Sprint/Project delay                  | 3                 | 5             | Ólafur            | Address the problem in the standup's and frequent reassessments. |
| Incompatibility of Translation APIs   | 3                 | 3             | Birkir            | Have fallback methods for each API, and make the system modular to easily swap out one service for another. |
| Classifier Model Inefficiency         | 3                 | 3             | Eysteinn          | Use baseline models for initial testing before using more complex models like BERT, roBERTa, and IceBERT. |
| Overfitting in Model Training         | 2                 | 4             | Birkir            | Utilize techniques such as cross-validation and dropout layers. |
| Illness in team                       | 2                 | 4             | Whole Team        | Cross-training and comprehensive documentation can help other team members pick up the slack. Tries not getting other team members sick. |
| API Rate Limiting or Costs            | 2                 | 3             | Birkir            | Caching translated data and batch processing could help in minimizing the number of API calls. |
| A team member quits                   | 1                 | 5             | Whole Team        | Having a documented and modular project architecture allows for easier transition of responsibilities. |
| External Dependency Failures (APIs down) | 1              | 2             | Whole Team        | Have a contingency plan, such as a local translation model otherwise wait and focus on a different task |
