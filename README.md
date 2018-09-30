# Counting Your Customers the Hard Way: Modeling Software-as-a-Serivce Usage and Churn with PyMC3

## Summary
Churn rate is an important metric for predicting the lifetime revenue of a customer in a Software-as-a-Serivce (SaaS) business. Similarly, a customer's usage of the SaaS product can give insight into the costs of servicing the customer. In this tutorial we show how to implement a probabilistic model of both churn and usage, using PyMC3, a popular probabilistic programming framework.


## Description
Churn rate---the rate at which customers stop patronizing a business---is an important metric for predicting the lifetime revenue of a customer in a Software-as-a-Serivce (SaaS) business. However, to forecast total lifetime value of the customer, the costs associated with the customer must be considered as well. While many models assume a fixed margin per customer, SaaS businesses usually have detailed metrics of a customer's usage of services, leading to a more accurate measure of cost.

In this tutorial we present a joint probabilistic model of both usage and churn, based on the work of Hardie and Ascarza. In this model, both usage and churn are functions of a latent process representing a customer's "commitment level." The commitment level is itself modeled using a hidden Markov model. Given a dataset of actual customer usage metrics, the parameters of the model are estimated using Monte-Carlo simulation.

We will give an overview of the model and demonstrate an implementation with the PyMC3, a popular probabilistic programming framework. We will show several insights to be gained from the model forecasts, including identifying customers at-risk for churn and predicting service capacity demand.


### Bibliography
E Ascarza, BGS Hardie. ["Modeling churn and usage behavior in contractual settings"](https://www0.gsb.columbia.edu/.../4587/ascarza_hardie_churn.pdf).

## Helpful Links
https://github.com/hstrey/Hidden-Markov-Models-pymc3
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
https://github.com/datascienceinc/pydata-seattle-2017
https://shopifyengineering.myshopify.com/blogs/engineering/how-shopify-merchants-can-measure-retention
