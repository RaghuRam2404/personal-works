# India Digital-Product Payments and Hosting Setup

## Direct recommendation

For 1Crore2Reps, the strongest practical launch stack is **Payhip + Razorpay + PayPal Business**, with settlements routed to the **current account** if its beneficiary name matches the legal business, proprietorship, or proprietor identity used during payment-provider KYC. Payhip can host eBooks, downloads, bundles, memberships and full courses; its free plan has all features and charges 5% per sale, while Razorpay adds Indian payment methods and PayPal adds a familiar route for overseas customers.[^1][^2][^3]

Do **not** start with Stan Store. Stan is attractive as a polished link-in-bio storefront, but India is absent from Stan’s current list of countries eligible for its managed Stripe Custom accounts. An India-based creator would therefore depend on PayPal Business, while still paying Stan $29 per month for Creator or $99 per month for Creator Pro.[^4][^5][^6]

The current-account name does **not automatically appear to buyers merely because that account receives settlements**. What buyers see depends on the storefront, checkout brand or billing descriptor: Payhip card statements normally show `PAYHIP*BUSINESSNAME`; Razorpay checkout can display the chosen brand name, while the buyer’s bank statement or SMS uses the billing name supplied during Razorpay activation; Lemon Squeezy shows `LEMSQZY*STORE` because it is the merchant of record.[^7][^8][^9]

## Recommended account stack

Create the accounts in this order:

1. **Dedicated business email** using the brand domain, used for every platform, receipt and support interaction.
2. **Payhip seller account** as the storefront and product-delivery layer.
3. **Razorpay merchant account** as the main Indian and card payment gateway.
4. **PayPal Business India account** as the second checkout route for international buyers.
5. **Optional Lemon Squeezy account** as a future merchant-of-record fallback for selected global products, not as the main course platform.
6. **Accounting folder or software** for invoices, payout reports, gateway fee invoices, refunds, FIRA/FIRC documents and monthly reconciliations.

Do not create parallel live stores on Stan, Gumroad, Lemon Squeezy and Payhip at launch. Multiple stores fragment links, analytics, customer records, refunds and support before demand is proven.

## Platform decision

| Platform | India payment fit | Products and courses | Starting economics | Tax role | Recommendation |
|---|---|---|---|---|---|
| **Payhip** | Connects to Razorpay in India and normally permits PayPal plus one other processor.[^10][^11] | Downloads, eBooks, courses, bundles, memberships and coaching; courses support video, text, quizzes, assignments, drip content and certificates.[^1][^12] | Free plan: 5% platform fee; Plus: $29/month + 2%; Pro: $99/month + 0%.[^1][^13] | Handles digital EU/UK VAT and US/Canadian sales tax, but is not presented as a universal merchant of record for all seller obligations.[^14] | **Best launch choice** for India-first payment flexibility and mixed product types. |
| **Stan Store** | India is not on Stan’s Stripe Custom country list, so an Indian seller would generally need PayPal Business.[^6][^4] | Digital downloads, courses, coaching and memberships in a polished link-in-bio experience.[^15][^5] | $29/month Creator or $99/month Creator Pro; Stan charges 0% transaction fee, while processor fees remain.[^16][^5] | Direct Stripe/PayPal processor model rather than a documented global merchant-of-record setup. | **Do not use now.** Monthly cost plus PayPal dependency is a weak fit before stable sales. |
| **Gumroad** | Simple seller onboarding and direct payouts where supported; buyer-facing bank information remains hidden.[^17][^18] | Digital products, eBooks, courses, bundles and recurring memberships.[^19][^20] | No monthly fee; current official help lists 10% + $0.50 for direct sales, with card processing or PayPal fees additional; Discover sales are 30%.[^21][^22] | Merchant of record and handles buyer sales-tax obligations worldwide.[^22] | **Fast fallback**, but expensive for $9–$27 offers. |
| **Lemon Squeezy** | India is supported, but bank payouts require a Stripe-approved account; without that approval, an Indian merchant must use PayPal payouts.[^23] | Digital files, video content, courses, subscriptions and software; secure file delivery is included.[^24] | Common base fee is 5% + $0.50, with possible additions for international payments and payout charges.[^25][^26][^27] | Merchant of record; buyer statement shows Lemon Squeezy’s descriptor.[^7][^28] | **Good global-tax fallback**, but less suitable for a rich native course experience and Indian payout simplicity. |

### Why Payhip wins

Payhip is the only option in this set that combines a no-monthly-fee launch, a native course player and direct Razorpay support for India. Razorpay through Payhip exposes cards, UPI, net banking and other locally available methods, while PayPal can normally be connected alongside another processor.[^11][^10]

Payhip’s free plan costs 5% of revenue. Compared only with Stan Creator’s $29 monthly platform fee and ignoring processor differences, Stan’s monthly fee equals 5% of approximately $580 monthly sales. In practice, the comparison is even less favourable at launch because Stan’s India setup would rely on PayPal, whereas Payhip can use Razorpay.[^5][^1]

## Payment architecture

### Main flow

The recommended customer flow is:

`Instagram/X → Payhip product page → Razorpay or PayPal → automatic Payhip delivery/course access → current-account settlement`

Payhip handles product pages, customer accounts, file delivery and course access. Razorpay or PayPal processes the money; Payhip is therefore not the settlement bank and the buyer never sees the current-account number.[^10][^1]

### Razorpay role

Razorpay supports Indian sellers and can accept buyers from more than 180 countries through its Payhip connection. The gateway offers cards, UPI, net banking, wallets and other payment methods, although the exact methods shown depend on the buyer’s region and merchant activation.[^10]

Razorpay settles all supported currencies into the Indian settlement account in INR. Domestic settlements normally follow the provider’s settlement cycle, with Razorpay documentation describing a standard T+2 working-day cycle for domestic payments after capture.[^29][^30]

International card acceptance is a separate activation step. Razorpay asks Indian applicants for business type, PAN, address and KYC information, and may request GSTIN or Udyam details where applicable; registered businesses with a valid website can apply for international cards and related methods.[^31]

Razorpay’s published international-card pricing is up to 3% per transaction, with GST applicable to the gateway fee; its international bank-transfer offer is advertised from 1%.[^32][^33]

### PayPal role

For India, use a **PayPal Business account**, not a personal PayPal account, when the main purpose is receiving export payments. PayPal explicitly advises unregistered operators to select “Business Individual” and registered operators to choose the applicable legal form.[^34]

Activation requires email confirmation, KYC, PAN, an Indian bank account and a purpose code. The bank-account name must correspond to the PAN-linked identity submitted during onboarding.[^35][^36]

PayPal India only supports receiving international payments, and its published standard fee for an international commercial receipt is 4.40% plus the currency-specific fixed fee. This makes PayPal useful as a checkout option, but generally not the cheapest primary gateway.[^37]

PayPal now provides downloadable weekly digital FIRA documentation for foreign inward remittances, which is useful for export and accounting records.[^38]

## Savings or current account

Use the **current account** if it belongs to the same proprietorship, business or proprietor identity used for the store and payment KYC. Current accounts are designed for frequent business collections and payments, while savings accounts are designed for personal saving and lower-volume personal transactions.[^39][^40]

Razorpay states that a government-registered business should provide the current account registered with that business. An unregistered operator or sole proprietor may provide personal bank details, but this is an onboarding allowance rather than a reason to mix growing business cash flow with personal savings.[^41]

Regular commercial activity through a personal savings account can conflict with the account’s intended use and may be flagged or restricted by the bank. A current account also makes revenue, expenses, refunds, taxes and profitability easier to reconcile.[^42][^43]

### Name-matching rule

Before connecting the current account, verify these three names:

- **Legal/KYC identity:** the individual, sole proprietorship, partnership or company registered with the gateway.
- **Bank beneficiary name:** the exact account-holder name on the cancelled cheque or statement.
- **Customer-facing brand:** `1Crore2Reps`, used on the storefront and checkout where permitted.

Razorpay requires the settlement-account beneficiary to precisely match the registered business name; for proprietorships it can also match the promoter’s PAN name. If the names do not match, Razorpay may require a new account or a bank letter confirming ownership.[^44]

Therefore:

- If the current account is in **1Crore2Reps**, the registered proprietorship name, or the same proprietor’s accepted name, use it.
- If the current account belongs to a different company, partnership or unrelated trade name, do not connect it until the KYC structure is aligned.
- If the savings account is the only name-matching account today, it may work for some sole-proprietor onboarding, but the current account should become the permanent settlement account once correctly aligned.[^41][^44]

## What buyers will see

The words “checkout name,” “statement descriptor,” “receipt name” and “bank beneficiary” describe different things. They should not be treated as one field.

| Buyer touchpoint | Likely visible name | Does current-account name show? |
|---|---|---|
| **Payhip storefront** | Store and product brand configured in Payhip | No |
| **Payhip card statement** | Typically `PAYHIP*BUSINESSNAME`; banks may format it differently.[^8] | Not merely because it is the payout account |
| **Razorpay checkout** | Brand name and logo configured in Checkout Styling; the checkout API also treats business/brand name as the displayed name.[^45][^9] | No, unless the same text was separately chosen as the checkout brand |
| **Razorpay buyer bank SMS/statement** | Billing name supplied during Razorpay account creation and activation; Razorpay says banks control this communication.[^9] | Possibly the same only if the KYC billing name and bank-account name match; the bank beneficiary itself is not directly exposed |
| **PayPal checkout/transaction** | PayPal business identity and email; card statement naming can be configured through PayPal subject to regional and bank rules.[^46][^47] | No direct exposure of the receiving bank account |
| **Gumroad purchase** | Creator profile name and support email; banking and payout information are not shown.[^18] | No, apart from rare processor-controlled exceptions documented by Gumroad |
| **Lemon Squeezy statement** | `LEMSQZY*STORE-ID` because Lemon Squeezy is merchant of record.[^7] | No |

### Practical privacy answer

If the current-account legal name is personal, customers will not normally see that name simply because funds settle there. However, a directly connected gateway may use the **KYC billing name** on the buyer’s card statement or bank SMS, even when the storefront shows `1Crore2Reps`; Razorpay explicitly distinguishes editable checkout branding from bank-controlled statement communication.[^9]

The safest approach is to set the customer-facing name to `1Crore2Reps` everywhere the platform permits, then make one real low-value purchase using a different card and UPI account. Check the checkout, payment-app confirmation, email receipt, pending statement and final settled statement because different banks can truncate or format descriptors differently.[^8][^46]

## Currency strategy

Payhip supports both USD and INR, but the store has a primary currency and Payhip requires that its currency match the connected Razorpay account’s currency configuration.[^48][^10]

Because the brand intends to price products in USD for Indians and NRIs, start the Payhip store in **USD** and activate international payments in Razorpay. Razorpay can charge supported foreign currencies and settle them in INR to the Indian bank account.[^30][^31]

However, do not advertise UPI until a live USD checkout test confirms that UPI is actually offered for the account and transaction configuration. Payment-method availability depends on region, currency and activation, and Payhip states that customers only see methods available in their region.[^10]

If USD checkout suppresses UPI for Indian buyers, use one of these controlled alternatives:

- Keep the main global Payhip product in USD and create a separate INR Razorpay Payment Page for Indian launches, with delivery handled manually or through automation.
- Temporarily price the entire Payhip store in INR if Indian buyers dominate early sales.
- Keep USD on Payhip and let Indian card/PayPal buyers pay internationally until order volume justifies a separate India-specific checkout.

Do not run two inconsistent public prices without explaining whether taxes and currency conversion are included.

## Tax and records

Payhip handles specified destination taxes—digital EU/UK VAT and US/Canadian sales tax—but this does not replace the seller’s Indian income-tax, GST, invoicing or export-record obligations.[^14]

Under India’s IGST framework, qualifying exports of services are zero-rated. The statutory conditions include an Indian supplier, an overseas recipient, an overseas place of supply, receipt in convertible foreign exchange and the parties not being merely establishments of the same person.[^49][^50]

Registered exporters using the route without payment of IGST generally furnish a Letter of Undertaking or bond under Rule 96A. Whether a platform sale of an eBook, course or template is treated as a service export, an OIDAR-type supply, a domestic sale, or a merchant-of-record transaction depends on the exact contract, buyer location, platform role and payment trail; a Chennai-based CA should review the final stack before launch.[^51][^52]

Retain these records monthly:

- Payhip orders, customer country and refunds.
- Razorpay payments, settlements and fee invoices.
- PayPal activity reports and weekly FIRA files.
- Platform invoices and foreign-exchange conversion records.
- Customer invoices or receipts.
- Refund and chargeback evidence.
- Bank statements and a reconciliation showing gross sales, tax, platform fees, processor fees, refunds and net deposits.

## Exact setup sequence

### Day 1: Align identity

- Decide the legal seller type: individual, unregistered business individual or sole proprietorship.
- Confirm the current-account beneficiary name.
- Ensure the chosen KYC identity, PAN and bank beneficiary can be accepted together.
- Use `1Crore2Reps` as the storefront brand, while entering legal identity accurately wherever the platform requests it.

### Day 2: Open Payhip

- Create the Payhip account with the dedicated business email.
- Set store name, support email and branding.
- Set the initial currency to USD.
- Add the refund policy, privacy policy, terms, contact page and product description.
- Create a hidden $1 or lowest-practical test product.

### Days 2–4: Activate Razorpay

- Create the Razorpay merchant account.
- Select the correct business structure.
- Complete PAN, Aadhaar/video KYC, address and bank verification.
- Submit the current account when its beneficiary matches the KYC identity.
- Activate international cards and select the appropriate purpose code.
- Configure Checkout Styling so the visible brand reads `1Crore2Reps`.[^9][^31]
- Generate the API key and secret, then connect Razorpay under Payhip’s Payment Details.[^10]

### Days 3–5: Activate PayPal

- Create PayPal Business, selecting Business Individual if unregistered or Sole Proprietorship if that is the actual structure.[^34]
- Confirm email, PAN/KYC, purpose code and bank account.[^36][^35]
- Configure the recognizable business or statement name where the India account allows it.
- Connect PayPal to Payhip as the second processor.[^53]

### Day 5: End-to-end test

Perform at least four tests:

1. Indian card purchase.
2. Indian UPI purchase, if offered.
3. PayPal purchase from an eligible international buyer/account.
4. Refund and access-revocation test.

For each test, record:

- Name shown on the storefront.
- Name shown in checkout.
- Name shown in the payment app.
- Receipt sender and receipt name.
- Pending bank-statement descriptor.
- Final bank-statement descriptor.
- Amount settled into the current account.
- Platform fee, gateway fee and tax on the fee.

### Launch gate

Do not publish the first paid-product link until all of these are true:

- A real payment succeeds.
- Automatic file or course access works.
- The buyer receives a recognizable receipt.
- The statement descriptor is acceptable.
- A refund works.
- The settlement account receives the correct net amount.
- Support, privacy, terms and refund pages are live.
- International payment activation is confirmed rather than assumed.

## Upgrade triggers

Stay on Payhip Free until its 5% fee materially exceeds a paid plan’s savings. Using platform fees alone, Plus at $29/month and 2% becomes cheaper than Free at roughly $967 monthly revenue because the 3-percentage-point saving equals $29; Pro at $99/month becomes cheaper than Free at roughly $1,980 monthly revenue because the 5-percentage-point saving equals $99.[^13][^1]

Consider Lemon Squeezy or Gumroad only when global tax administration and merchant-of-record simplicity are worth more than the lost margin, payment-method limitations or weaker course experience. Consider Stan only if its link-in-bio conversion tools become demonstrably valuable and an India-compatible processor setup improves; as of September 2026, Stan’s managed Stripe country list still excludes India.[^6]

## Final configuration

- **Storefront and delivery:** Payhip Free.
- **Primary gateway:** Razorpay.
- **Secondary gateway:** PayPal Business India.
- **Settlement account:** Current account, provided its beneficiary matches the registered seller or accepted proprietor identity.
- **Public brand:** 1Crore2Reps.
- **Launch currency:** USD, subject to a live UPI availability test.
- **Stan Store:** Do not open now.
- **Gumroad:** Emergency fast-launch fallback only.
- **Lemon Squeezy:** Optional merchant-of-record fallback for selected global products.
- **Pre-launch professional check:** One focused CA review covering GST registration, export classification, LUT, invoicing and reconciliation for the chosen legal structure.

---

## References

1. [FAQ - Payhip](https://payhip.com/faq) - These are the most common questions we get asked when selling digital downloads and memberships on P...

2. [Payhip - Create a free website and sell online](https://payhip.com/) - Payhip is the easiest way to sell digital downloads and courses. No technical skills required. Creat...

3. [Ecommerce Payment Gateways for Digital Products](https://payhip.com/payment-gateways) - Payhip connects directly with Stripe, PayPal, Square, Mollie, Mercado Pago, Flutterwave, Razorpay, a...

4. [How to Connect Stan with PayPal](https://help.stan.store/article/53-how-to-connect-stan-with-paypal) - Want to get paid through PayPal? You’re just a few clicks away! Make sure your account’s ready, then...

5. [Stan Store vs Stanley](https://help.stan.store/article/438-stan-store-vs-stanley) - Confused between Stan Store and Stanley? Here’s a quick breakdown of what each one is and how they’r...

6. [Countries Available for Stripe Custom Accounts - Stan Store ...](https://help.stan.store/article/217-countries-available-for-stripe-custom-accounts) - When you connect Stripe to your Stan account, you’ll be prompted to set up a Stripe Custom account t...

7. [LEMSQZY* Statement Descriptor](https://docs.lemonsqueezy.com/help/getting-started/statement-descriptor) - When your customers purchase through your store they will see LEMSQZY*STORE charge on their bank sta...

8. [Unrecognized Payhip Charges - Help Center](https://help.payhip.com/article/385-unrecognized-payhip-charges) - If you've noticed a Payhip charge on your bank or card statement, it most likely means you made a pu...

9. [Checkout Styling | Razorpay Docs](https://razorpay.com/docs/payments/dashboard/account-settings/checkout-styling/?preferred-country=IN)

10. [Connect Your Razorpay Account - Payhip Help](https://help.payhip.com/article/368-connect-your-razorpay-account) - Razorpay supports payment processing for sellers in India. Sellers in India can accept payments from...

11. [How Do I Get Paid?](https://help.payhip.com/article/173-how-do-i-get-paid) - Before you can sell through Payhip, you will need to connect to one of our supported payment process...

12. [Sell Courses Online For Free](https://payhip.com/features/sell-courses) - Setup and sell courses online with Payhip to make them look incredible. Create different lesson type...

13. [Selar Alternative](https://payhip.com/selar-alternative-ad) - Selar Alternative. You focus on creating and we'll handle all the tech for you. Get started for free...

14. [Sales Tax & VAT - Payhip Features](https://payhip.com/features/vat-taxes) - We report and pay digital EU VAT and UK VAT on your behalf. You can also setup your own sales taxes.

15. [How to Turn What You Know Into a Course on Stan Store](https://stan.store/blog/stan-store-online-course-success/) - Learn how to create an online course on Stan Store in this guide: customize your offer, build module...

16. [Stan, Stripe, and PayPal Transaction Fees](https://help.stan.store/article/83-stan-stripes-transaction-fees) - One of our values at Stan is being Creator First. One way we embody this is by not taking a cut of y...

17. [Filling out payout settings - Gumroad Help Center](https://gumroad.com/help/article/260-your-payout-settings-page.html) - Requirements: If you are 18 or older, you fill out these settings yourself. If you are 13 to 17, you...

18. [Protecting creator privacy - Gumroad Help Center](https://gumroad.com/help/article/120-protecting-your-privacy-on-gumroad) - In this article: What information do your customers see PayPal Stripe Exceptions What information do...

19. [Adding a product - Gumroad Help Center](https://gumroad.com/help/article/149-adding-a-product.html) - In this article: Watch a walkthrough Choose a product type Set a price Describe your product Provide...

20. [Gumroad features: Sell digital products, memberships & courses](https://gumroad.com/features) - Everything you need to sell online: instant digital delivery, memberships, courses, pay-what-you-wan...

21. [Gumroad's fees - Gumroad Help Center](https://gumroad.com/help/article/66-gumroads-fees) - Gumroad's fees are simple For sales made on Gumroad's website, we charge a 10% + $0.50 fee per trans...

22. [Gumroad pricing: 10% + 50¢ direct, 30% via Discover](https://gumroad.com/pricing) - Direct sales: 10% + 50¢ per sale, $0 monthly. Discover marketplace sales: 30%. Gumroad handles sales...

23. [Supported Countries - Lemon Squeezy Docs](https://docs.lemonsqueezy.com/help/getting-started/supported-countries) - If you are a merchant from India without an approved Stripe account, you will have to use PayPal for...

24. [Sell Digital Products • Lemon Squeezy](https://www.lemonsqueezy.com/ecommerce/digital-products) - Sell digital products any way you want. If it’s digital, we can help you sell and distribute it arou...

25. [Buildspace](https://www.lemonsqueezy.com/buildspace) - Taking part in a Buildspace event and want to add payments to your idea? Lemon Squeezy is a ready-ma...

26. [Docs: Sales Tax and VAT • Lemon Squeezy](https://docs.lemonsqueezy.com/help/payments/sales-tax-vat) - Lemon Squeezy is known as the merchant of record for all sales through our platform. That means we t...

27. [Payout fees](https://docs.lemonsqueezy.com/help/getting-started/fees) - Get familiar with the fees associated with using Lemon Squeezy.

28. [Payments, Tax & Subscriptions for SaaS](https://www.lemonsqueezy.com/) - Sell digital products and SaaS software the easy peasy way with Lemon Squeezy. As your merchant of r...

29. [About Settlements | Razorpay Docs](https://razorpay.com/docs/payments/settlements/?preferred-country=IN)

30. [Currency Conversion - Razorpay](https://razorpay.com/docs/payments/international-payments/currency-conversion) - Know how currency conversion works in the APIs, Check the supported international currencies. Watch ...

31. [About International Payments](https://razorpay.com/docs/payments/international-payments) - You can accept payments from your customers in more than 160+ foreign currencies using our Payment G...

32. [Accept International Payments for Global Business](https://razorpay.com/accept-international-payments/) - Success Rate. 90-95% ; Coverage. 135 currencies, global cards, Apple Pay, Google Wallet* & bank tran...

33. [Accept international payments effortlessly with multi ...](https://razorpay.com/international/) - Accept International payments from Credit cards, Debit cards and PayPal wallet. Supports nearly 100 ...

34. [Which account type should I choose, Personal or Business? - PayPal](https://www.paypal.com/in/cshelp/article/which-account-type-should-i-choose-personal-or-business-help1399) - Explains account types: Personal or Business

35. [How do I receive payments through PayPal?](https://www.paypal.com/in/cshelp/article/how-do-i-receive-payments-through-paypal-help667) - Confirm email and identity for PayPal payments in India. Verify PAN, add bank account, and select pu...

36. [How to activate your Business Account](https://www.paypal.com/in/brc/article/activate-your-business-account) - All the steps you need to know about Business Account activation: confirm your email, verify your id...

37. [PayPal Fees for Sellers | PayPal IN](https://www.paypal.com/in/business/paypal-business-fees) - Everything sellers in India need to know about PayPal's charges and rates for each transaction, char...

38. [FIRC (Foreign Inward Remittance Certificate) - PayPal India](https://www.paypal.com/in/business/firc-certificate) - Get more information on requesting Foreign Inward Remittance Certificates (FIRC) for your inward rem...

39. [Current Account](https://razorpay.com/x/current-accounts/) - Open RazorpayX Current Account to manage business transactions. Optimize your money movement at ease...

40. [Current Account FAQs - ICICI Bank](https://www.icici.bank.in/business-banking/accounts/current-account/current-account-faqs) - Find answers to common questions about current account at ICICI Bank. Learn features, eligibility, c...

41. [Account Activation Support](https://razorpay.com/docs/payments/account-activation-support/?preferred-country=IN) - List of commonly used terms and details about certificates related to Razorpay Account activation.

42. [Current Account for Freelancers & Sole Proprietors](https://www.utkarsh.bank.in/blogs/current-account-for-freelancers-and-sole-proprietors-do-you-actually-need-one) - Open savings account & current accounts, invest in fixed deposits, apply for loans, and enjoy secure...

43. [Complete Guide to Business Current Account | Bank of Baroda](https://bankofbaroda.bank.in/banking-mantra/savings/articles/complete-guide-to-business-current-account) - Get the complete guide to business current account. Know what they are, why they are important for y...

44. [Bank Account Details | Razorpay Docs](https://razorpay.com/docs/payments/dashboard/account-settings/bank-account-details/?preferred-country=IN) - Steps to update your bank account details on the Razorpay Dashboard.

45. [Standard Checkout Integration Guide - Razorpay](https://razorpay.com/docs/developer-tools/integrations/standard-checkout/?preferred-country=IN)

46. [Descriptors](https://developer.paypal.com/braintree/articles/control-panel/transactions/descriptors) - Our mission is to empower developers with the tools, resources, and simple-to-use SDKs and APIs to b...

47. [How to update your business name on customers' credit ...](https://www.paypal.com/us/brc/article/how-to-update-merchant-name-for-customers-credit-card-statements) - Learn how to update your business name on customers' credit card statements to prevent consumers fro...

48. [Store Language and Currency - Help Center](https://help.payhip.com/article/234-store-language-and-currency) - In this article, we will cover the following topics: Where can I change the store language? What lan...

49. [Section 16. Zero rated supply.](https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/acts/2017_IGST_Act/active/chaptervii/section16_v1.00.html)

50. [[PDF] Circular No. 161/17/2021-GST](https://cbic-gst.gov.in/pdf/Circular-No-161-14-2021-GST.pdf) - “export of services” means the supply of any service when,–– (i) the supplier of service is located ...

51. [[PDF] Circular No. 37/11/2018-GST](https://cbic-gst.gov.in/pdf/circularno-37-cgst.pdf) - 4. Exports without LUT: Export of goods or services can be made without payment of integrated tax un...

52. [Rule 96A - CBIC Tax Information](https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/rules/cgst_rules/active/chapter10/rule96a_v1.00.html)

53. [Connect Your PayPal Account - Help Center - Payhip](https://help.payhip.com/article/64-connecting-your-paypal-account) - Payment Gateways - Stripe, Paypal, Square, Mercado Pago, Mollie, Paystack In this article: General I...

