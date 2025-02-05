# Case description

A company in the retail food sector conducted a pilot marketing campaign targeting 2,240
randomly selected customers with a new product offer. The results showed a low success
rate (15%) and a negative profit. To enhance future campaigns, the company
seeks to develop a predictive model that will:

- Analyze customer characteristics to understand what influences purchasing decisions.
- Segment customers based on their behavior and likelihood to respond.
- Improve targeting efficiency by selecting only high-potential customers, making the next campaign profitable.

This project was inspired by [this
repository](https://github.com/ifood/ifood-data-business-analyst-test). I did not
participate in the selection process described in the linked repository, nor do I have
any association with the company mentioned there. I simply came across the project
online, considered it interesting, and decided to add it to my portfolio.

The full case description can be found in the PDF file [case_description.pdf](./case_description.pdf). This is the original case description from the repository that inspired this project. Again, I have no affiliation with the company mentioned in the case description.

## Data Dictionary

Personal Data

- `ID`: Unique customer identifier
- `Year_Birth`: Customer's year of birth
- `Education`: Customer's education level
- `Marital_Status`: Customer's marital status
- `Income`: Customer's annual family income
- `Kidhome`: Number of children in the customer's household
- `Teenhome`: Number of teenagers in the customer's household
- `Dt_Customer`: Date the customer enrolled with the company
- `Recency`: Number of days since the customer's last purchase
- `Complain`: 1 if the customer has complained in the last 2 years, 0 otherwise

Product Data

- `MntWines`: Amount spent on wine in the last 2 years
- `MntFruits`: Amount spent on fruits in the last 2 years
- `MntMeatProducts`: Amount spent on meat in the last 2 years
- `MntFishProducts`: Amount spent on fish in the last 2 years
- `MntSweetProducts`: Amount spent on sweets in the last 2 years
- `MntGoldProds`: Amount spent on gold products in the last 2 years

Campaign Data

- `NumDealsPurchases`: Number of purchases made with a discount
- `AcceptedCmp1`: 1 if the customer accepted the offer in the 1st campaign, 0 otherwise
- `AcceptedCmp2`: 1 if the customer accepted the offer in the 2nd campaign, 0 otherwise
- `AcceptedCmp3`: 1 if the customer accepted the offer in the 3rd campaign, 0 otherwise
- `AcceptedCmp4`: 1 if the customer accepted the offer in the 4th campaign, 0 otherwise
- `AcceptedCmp5`: 1 if the customer accepted the offer in the 5th campaign, 0 otherwise
- `Response`: 1 if the customer accepted the offer in the last (pilot) campaign, 0 otherwise (*target*)

Purchase Location Data

- `NumWebPurchases`: Number of purchases made through the company's website
- `NumCatalogPurchases`: Number of purchases made using a catalog
- `NumStorePurchases`: Number of purchases made directly in stores
- `NumWebVisitsMonth`: Number of visits to the company's website in the last month
