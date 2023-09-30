# Risk Analysis for Sentiment-Analysis in Icelandic using Machine Translation

| Risk | Likelihood (1-5) | Impact (1-5) | Impact (Team Members) | Responsibility | Mitigation Strategy |
| --- | --- | --- | --- | --- | --- |
| Incompatibility of Translation APIs | 3   | 3   | Whole Team | Whole Team | Have fallback methods for each API, and make the system modular to easily swap out one service for another. |
| Classifier Model Inefficiency | 3   | 4   | Whole Team | Whole Team | Use baseline models for initial testing before using more complex models like BERT, roBERTa, and IceBERT. |
| Overfitting in Model Training | 3   | 1   | Whole Team | Whole Team | Utilize techniques such as cross-validation and dropout layers. |
| Resource Constraints (Time and Computing Power) | 3   | 4   | Whole Team | Project Manager | Prioritize key features and models that are critical to the project. Consider using cloud computing resources. |
| Sprint/Project delay | 3   | 5   | Whole Team | Project Manager | Implement Agile methodologies for better time management and frequent reassessments. |
| API Rate Limiting or Costs | 2   | 3   | Whole Team | Whole Team | Caching translated data and batch processing could help in minimizing the number of API calls. |
| External Dependency Failures (Server downtimes for APIs) | 2   | 3   | Whole Team | Whole Team | Have a contingency plan, such as a local translation model or other fallback options. |
| Training stops/Computer crashes | 4   | 4   | Whole Team | Team member in question | Regular backups and distributed training could mitigate this risk. |
| Illness in team | 2   | 4   | Whole Team | Whole Team | Cross-training and comprehensive documentation can help other team members pick up the slack. Tries not getting other team members sick. |
| A team member quits | 2   | 5   | Whole Team | Project Manager | Having a documented and modular project architecture allows for easier transition of responsibilities. |
