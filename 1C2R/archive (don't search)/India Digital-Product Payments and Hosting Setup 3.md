# 1Crore2Reps Commerce Stack: Git-Based Website, Newsletter, Dodo Payments and Discord

## Executive decision

The technically clean stack is **GitHub for source control, Cloudflare Pages for commercial website hosting, Kit for the newsletter, and Discord for the community**. Dodo Payments can provide checkout, subscriptions, downloadable-file fulfilment and automated Discord access, but it should be adopted only after Dodo gives written pre-clearance for the exact 1Crore2Reps products.

That pre-clearance is not a minor formality. Dodo’s current Merchant Acceptance Policy welcomes ordinary digital goods and templates, but explicitly excludes unlicensed financial tools, investment strategies, wealth-building courses, and the full health-and-wellness product class, including weight-loss programs. E-books also receive additional review, and Dodo monitors whether gated Discord content matches the product disclosed at onboarding. A combined money-and-muscle brand therefore sits unusually close to two prohibited categories even if the content is responsible and educational.[^1]

The second correction concerns hosting. GitHub Pages technically supports custom domains, but GitHub’s current terms say Pages is not intended or allowed as free hosting for an online business or ecommerce site primarily directed at commercial transactions. The repository can remain on GitHub, while Cloudflare Pages deploys it commercially from the same repository on every push.[^2][^3][^4][^5]

## Revised architecture

| Layer | Recommended service | Responsibility |
|---|---|---|
| Source control | GitHub private repository | Version control, pull requests and content history |
| Website hosting | Cloudflare Pages | Public website, product pages, legal pages and newsletter archive links |
| Domain and DNS | Cloudflare DNS or current registrar | Apex domain, `www`, email records and optional newsletter subdomains |
| Newsletter | Kit | Forms, lead-magnet delivery, broadcasts and email sequences |
| Checkout and billing | Dodo Payments, conditional on written approval | One-time payments, recurring subscriptions, invoices, refunds and customer portal |
| Digital delivery | Dodo entitlements | PDFs, trackers, templates and bundles |
| Community | Discord | Conversation, accountability, challenges and member-only resources |
| Community access control | Dodo Discord entitlement | Grant and revoke the paid-member role based on payment status |
| Banking | Matching Indian current account | Receive business payouts and keep business reconciliation separate |
| Course platform | None initially | Add only after repeated buyer demand proves the syllabus |

This structure separates assets that should remain portable. The website, mailing list, source files and customer-support records stay under the brand’s control; Dodo can be replaced if its policies, fees or product fit change. Discord is likewise a separate service—not a community hosted inside Dodo.

## Website hosting correction

### Do not use GitHub Pages

A custom domain and CNAME make GitHub Pages technically capable of displaying the site, but they do not change GitHub’s usage policy. GitHub says Pages is primarily for static pages showcasing projects and is not intended or allowed as free hosting for an online business, ecommerce site or site mainly facilitating commercial transactions.[^3][^2]

A site containing product pages, “Buy” buttons and paid memberships would be difficult to characterize as anything other than commercial. Relying on GitHub Pages would create avoidable platform risk precisely when sales begin to work.

### Use GitHub plus Cloudflare Pages

Cloudflare Pages connects directly to a GitHub repository and automatically rebuilds and deploys the website after changes are pushed. Its published free tier includes 500 builds per month, custom domains, unlimited sites, static requests and bandwidth, making it suitable for an early-stage static commercial site.[^6][^4][^5]

Recommended deployment:

1. Keep the website in a private GitHub repository.
2. Connect that repository to Cloudflare Pages.
3. Configure `main` as the production branch.
4. Add the apex domain and `www` to the Cloudflare Pages project.
5. Redirect one hostname consistently to the other.
6. Retain preview deployments for changes before production.

For an apex domain, Cloudflare requires the domain to be a Cloudflare zone with its nameservers pointed to Cloudflare. For a subdomain, a CNAME can point to the project’s `pages.dev` hostname. This preserves the same Git-based workflow while removing the GitHub Pages commercial-use conflict.[^7]

## Proposed domain map

| Hostname or path | Purpose | Service |
|---|---|---|
| `1crore2reps.com` | Main website | Cloudflare Pages |
| `www.1crore2reps.com` | Redirect to apex | Cloudflare Pages/redirect rule |
| `/products` | Product catalogue | Static site |
| `/products/product-name` | Long-form sales page | Static site |
| `/newsletter` | Newsletter explanation and signup | Static site with Kit form |
| `/articles/...` | Searchable authority content | Static site |
| `/community` | Discord membership sales page | Static site |
| `/legal/privacy` | Privacy notice | Static site |
| `/legal/terms` | Website and product terms | Static site |
| `/legal/refunds` | Refund policy | Static site |
| `/checkout/success` | Post-purchase instructions | Static site |
| `pages.1crore2reps.com` | Optional Kit-hosted landing pages | Kit |
| `support@1crore2reps.com` | Customer support | Existing email provider |

Kit advises using a subdomain for its landing pages when the main domain already hosts a website. Kit forms can also be embedded directly in any site accepting JavaScript or HTML; its JavaScript embed is the recommended method. For this architecture, embedded forms maintain the strongest visual consistency, while `pages.` remains available for campaign landing pages.[^8][^9][^10]

Configure Kit as a verified sending domain. Its verification uses SPF, DKIM and DMARC records to authorize branded sending and improve mailbox trust. Kit states those authentication records can coexist with the top-level website configuration.[^11][^12]

## Dodo’s actual role

Dodo is not the website host or the Discord community. It is the **transaction and entitlement layer** between them.

For ordinary approved products, the flow is:

> Website sales page → Dodo-hosted checkout → successful payment → digital file or Discord entitlement → current-account payout

As Merchant of Record, Dodo becomes the legal seller for the customer transaction, appears on relevant receipts or statements, calculates and remits applicable transaction taxes, handles payment disputes, and pays the creator the net proceeds. The creator remains responsible for Indian income tax and other business obligations attached to those payouts.[^13]

Dodo publishes no mandatory monthly fee. Its standard schedule lists 4% plus $0.40 for domestic US transactions, an additional 1.5% for international payments, 0.5% for subscriptions, and 4% plus $0.15 for domestic Indian INR cards and UPI, subject to its current full fee schedule.[^14]

### Static-site checkout

The first implementation should use static payment links behind website buttons. Dodo documents product links in the format `https://checkout.dodopayments.com/buy/{product_id}`, with a return URL sending the buyer back to the website.[^15]

This approach is preferable for launch because:

- It requires no backend.
- No secret API key is exposed in browser code.
- Dodo hosts the sensitive checkout.
- It works with a fully static Cloudflare Pages site.
- Each product can have a dedicated button and tagged return URL.

Do not put a Dodo secret API key in the GitHub repository or frontend JavaScript. Dodo’s dynamic checkout sessions and overlay require a server-side endpoint to create a secure session; a static host alone cannot make that secret-bearing call safely. If an overlay or inline checkout becomes important later, add a small Cloudflare Worker to create sessions and validate requests.[^16]

Dodo also states that its hosted checkout cannot simply be placed inside a generic iframe. Use the hosted redirect, the official overlay, or the official inline-checkout mechanism instead.[^17]

## Discord integration clarified

### Yes, Discord can connect directly

Dodo now provides a native Discord entitlement. A buyer completes checkout, connects a Discord account through an OAuth link in the delivery email or customer portal, and Dodo’s bot adds the buyer to the chosen server or finds the existing member and assigns the configured role.[^18][^19]

When a subscription is cancelled, paused, expires, changes plan or is refunded, Dodo revokes the role according to the payment lifecycle. This removes the need for Zapier or a custom role-management bot for the basic membership case.[^20][^21]

However, “Discord is part of Dodo” should be understood precisely:

- The Discord server still lives on Discord.
- All channels, moderation, rules and content remain the creator’s responsibility.
- Dodo handles payment status and access entitlement.
- Members must authorize their Discord identity through Dodo’s OAuth flow.
- Dodo’s bot needs server permissions including role management and member access.[^19]
- Dodo can revoke the paid role, but it does not operate the community or create recurring value.

### Minimum role structure

Use a small role system:

| Role | Access |
|---|---|
| `@everyone` | Rules, welcome and public announcements only |
| `@Free Member` | Optional public discussion and newsletter prompts |
| `@1C2R Member` | Paid channels, challenges, templates and accountability |
| `@Moderator` | Message and thread moderation without full administration |
| `@Owner` | Full control |
| Dodo bot role | Only permissions necessary to invite members and manage the paid role |

Discord cautions that the Administrator permission grants every permission and bypasses channel restrictions, so it should be assigned only when absolutely necessary. Position the Dodo bot role above `@1C2R Member` in the hierarchy so it can assign that role, but do not give the bot broader control than required. Discord’s role hierarchy permits a role to manage only roles positioned below it.[^22][^23]

Configure all paid channels as private and expose them only to `@1C2R Member`. Discord supports role-level and channel-level permissions, and its “View Server As Role” feature can be used to test that non-paying members cannot see paid channels.[^24]

### Membership fulfilment flow

1. Visitor reads the community sales page.
2. Visitor clicks **Join the Membership**.
3. Dodo processes the recurring subscription.
4. Dodo emails the receipt and Discord connection link.
5. Buyer authorizes Discord access.
6. Dodo assigns `@1C2R Member`.
7. Discord’s welcome channel directs the member to the start-here checklist.
8. On cancellation, expiry, refund or applicable failed-renewal state, Dodo removes the role.[^21][^18]

Create a manual support procedure for buyers who use the wrong Discord account, fail OAuth or leave the server. Dodo’s entitlement states include pending, delivered, failed and revoked, which should be checked before promising that access was completed. If custom automations are later added, treat `entitlement_grant.delivered`, rather than only `payment.succeeded`, as the fulfilment source of truth.[^25][^21]

## Critical Dodo policy risk

This is the largest change to the earlier recommendation.

Dodo’s current policy explicitly prohibits:

- Unlicensed financial tools.
- Investment strategies.
- Wealth-building courses.
- Tax calculators.
- Financial or tax planning and other regulated advice.
- Health and wellness products as a class, including weight-loss programs.
- Miracle, misleading or unverifiable money and fitness claims.[^1]

It also places e-books and written digital publications under enhanced review, even when their subject matter is otherwise acceptable. Dodo reviews not only the sales page but also what is delivered through Discord, files and other gated channels; material mismatch can trigger review, restriction or closure.[^26][^1]

The initial “Double Compounding Starter System” combines money and fitness. Depending on its content, Dodo could interpret it as a wealth-building or health-and-wellness product. The Panda, brand style, an educational disclaimer and careful wording do not override category policy.

### Pre-clearance requirement

Before building the checkout, send Dodo compliance:

**Subject:** Pre-clearance request — educational habit tracker and paid accountability community

> Hello Dodo Compliance,
>
> I operate an Indian sole proprietorship under the brand 1Crore2Reps. The brand publishes general educational content for Indians and NRIs about personal financial habits and physical-training consistency.
>
> The proposed products are downloadable planning tools and accountability systems: budgeting and savings-habit trackers, workout-consistency trackers, weekly review templates, and a Discord community focused on implementation and peer accountability.
>
> They will not provide individualized investment advice, securities recommendations, return guarantees, tax advice, medical diagnosis, treatment, supplements, weight-loss claims, personalized coaching, or custody/management of customer funds. The same restrictions will apply inside the Discord community.
>
> Please confirm in writing whether Dodo can approve: (1) these downloadable products, (2) a combined money-and-fitness habit bundle, and (3) recurring Discord community access. A draft sales page, sample files and community channel plan can be provided for review.

Send the exact draft product and sample content—not only a broad description—to `compliance@dodopayments.com`. Dodo expressly recommends pre-contact for uncertain categories and states that e-books may require disclaimers, demo access or policy links.[^1]

### Decision rule

| Dodo response | Action |
|---|---|
| Written approval for both pillars and Discord | Use Dodo for products and membership, preserving the approval email and submitted materials |
| Approval for neutral productivity/habit products only | Sell only the approved scope through Dodo; keep finance and fitness content free until another processor is verified |
| Approval for one pillar only | Split the commercial catalogue and never route unapproved products or Discord channels through Dodo |
| Ambiguous or verbal approval | Request written confirmation tied to the exact URLs and files |
| Decline | Use the fallback stack; do not disguise or misclassify the products |

Misclassifying the business to bypass Dodo’s restrictions can lead to suspension, halted transactions, withheld payouts, refunds and other enforcement. The correct strategy is platform fit, not semantic camouflage.[^1]

## Fallback commerce stack

If Dodo declines the product category, retain the website, domain, Kit and Discord. Replace only the transaction and fulfilment layer.

The practical fallback is **Payhip plus Razorpay for India**, with PayPal or another approved global option where appropriate. Payhip supports digital products, courses, memberships and bundles; it has direct integrations with Razorpay and PayU for Indian sellers. This route does not provide the same Merchant-of-Record allocation for every transaction, so Indian GST, export treatment, invoicing and foreign-remittance documentation must be reviewed with an Indian CA before launch.[^27][^28][^29]

Payhip does not need to replace the branded website. Product pages can remain on the static site and link to Payhip checkout or storefront pages. Discord role automation would require Zapier or a custom flow rather than Dodo’s native entitlement; Zapier lists Payhip-to-Discord workflows, but cancellation and role revocation should be tested rather than assumed.[^30][^31]

The fallback should be configured only if required. Running Dodo, Payhip and Razorpay simultaneously at launch would fragment customer records, refunds and analytics.

## Newsletter operating model

The newsletter should be the trust engine, not a separate product at launch. Kit’s free plan currently provides up to 10,000 subscribers, unlimited forms, landing pages and broadcasts, although a Kit-managed recommendation slot applies on that plan.[^32][^33]

Recommended weekly issue:

1. **One Money System:** A general habit, framework or checklist without individualized investment instruction.
2. **One Muscle System:** A consistency, scheduling or training-log principle without diagnosis, treatment or health claims.
3. **The Double Rep:** One action to complete in both pillars during the next seven days.
4. **Tool of the Week:** A useful extract from a paid tracker or template.
5. **CTA:** One focused action—download the free reset, buy the current product or join a waitlist.

A free newsletter is more valuable initially than a paid newsletter because it owns the audience relationship and provides repeated evidence of quality. Introduce a paid newsletter only if subscribers repeatedly ask for research or analysis that is valuable every month and falls safely within processor policy. Otherwise, monetize through concrete systems, trackers and memberships.

## Product and membership sequence

### Stage 1: Audience and list

Publish the free newsletter and create a lead magnet that demonstrates the operating-system approach. The objective is not raw subscriber count; it is tagged intent—money systems, muscle systems or combined accountability.

### Stage 2: One-time product

Launch one focused product only after processor pre-clearance. The deliverable should produce an immediate practical outcome and contain:

- A short implementation guide.
- A tracker or dashboard.
- Weekly review cards.
- A restart protocol.
- Clear version and support information.

Avoid guaranteeing wealth, returns, body transformation or health outcomes. Use evidence-based, bounded promises such as “build a repeatable weekly review” or “track 30 days of implementation.”

### Stage 3: Cohort challenge

Run a fixed 21- or 30-day challenge for existing customers. If Dodo approves the category, it can be sold as a one-time product that also grants a temporary Discord role. The challenge tests whether members participate, help each other and value accountability without prematurely creating an indefinite membership.

### Stage 4: Recurring Discord membership

Open recurring membership only after the challenge demonstrates demand. A minimum readiness gate should include:

- At least 20 paying product customers.
- At least 10 active challenge participants.
- At least five explicit requests for ongoing accountability.
- A four-week content and moderation calendar prepared in advance.
- One clear recurring outcome beyond “chat access.”

The membership needs a repeatable service loop:

- Monday money-and-muscle planning thread.
- Midweek progress check.
- Friday review and scoreboard.
- Monthly challenge.
- New template or tool drop.
- Member wins and implementation clinic conducted as text or screen-share without requiring a face.

This keeps the community faceless while making the value depend on systems and accountability rather than personality.

### Stage 5: Course later

Create a course only when support questions reveal a stable curriculum. A faceless course can use slides, screen recordings, annotated spreadsheets, voiceover and downloadable exercises. The course platform is therefore a later fulfilment decision, not a current infrastructure dependency.

## Recommended Discord design

Keep the initial server intentionally small:

| Category | Channel | Purpose |
|---|---|---|
| Start | `#start-here` | Setup checklist and community promise |
| Start | `#rules-and-scope` | No personalized investment, tax, medical or treatment advice |
| Start | `#announcements` | Product updates and challenge schedule |
| Action | `#weekly-plan` | Monday commitments |
| Action | `#money-reps` | Budgeting, saving and implementation habits |
| Action | `#muscle-reps` | Training consistency and tracking habits |
| Action | `#weekly-review` | Friday reflection and scorecard |
| Resources | `#tool-library` | Approved templates and current versions |
| Social proof | `#wins` | Verifiable implementation wins without income/body promises |
| Support | `#help-desk` | Product and access support |

Do not begin with dozens of empty channels. Paid community value should come from a predictable operating rhythm, not server complexity.

## Data ownership and automation

Maintain a small master customer ledger outside every platform. At minimum record:

- Dodo customer ID.
- Email address.
- Product purchased.
- Payment or subscription ID.
- Entitlement status.
- Discord username/ID when available.
- Kit subscriber ID or tag.
- Purchase date, refund status and support history.

Do not expose Dodo API secrets in a static repository. For future server-side automation, store secrets in Cloudflare Worker environment variables and verify Dodo webhook signatures before changing access. The static frontend should contain only public product IDs and hosted checkout URLs.

Useful later automations include:

- Tag a customer in Kit after an approved purchase.
- Send a buyer-specific onboarding sequence.
- Remove promotional emails for products already owned.
- Notify support when a Discord entitlement fails.
- Trigger retention email for `subscription_on_hold` rather than treating it like an intentional cancellation; Dodo distinguishes failure-related holds from manual or cancelled revocations.[^21]

## Banking and tax operations

Use the current account when its beneficiary identity can be tied cleanly to the verified sole proprietor and Udyam records. The account name does not become the storefront brand merely because it receives payouts; Dodo, as Merchant of Record, is the legal transaction seller and appears on customer transaction records, while the business receives net payouts.[^13]

Retain monthly:

- Order export.
- Refund and chargeback export.
- Dodo fee/remittance statement.
- Payout report.
- Bank credit confirmation.
- Currency-conversion records.
- Product-level revenue split.

Dodo’s MoR role covers customer-transaction taxes in its MoR flow, not the proprietor’s Indian income tax or every domestic business obligation. Ask an India-based CA to review the Dodo agreement, the characterization of payouts, GST registration position, export documentation and bookkeeping before volume becomes material.[^13]

## Ninety-day extension

### Days 1–7: Infrastructure and clearance

- Put the website code in a private GitHub repository.
- Deploy it through Cloudflare Pages.
- Connect the custom domain and enforce HTTPS.
- Create support, privacy, terms and refund pages.
- Write the exact first product specification.
- Send Dodo the compliance pre-clearance request with sample pages and files.
- Do not publish a Dodo checkout until written scope is clear.

### Days 8–14: Newsletter foundation

- Create the Kit account and verified sending domain.
- Embed the Kit form on the website.
- Create a lead magnet and automated delivery email.
- Draft a three-email welcome sequence.
- Publish the first two newsletter issues.
- Tag subscribers by money, muscle and combined interests.

### Days 15–28: Product completion

- Finish one pre-cleared product.
- Test every file and external link.
- Create the product sales page and FAQ.
- Add version, support and refund information.
- If Dodo approves, create the product and attach file entitlements.
- If Dodo declines, activate the Payhip/Razorpay fallback after CA review.

### Days 29–35: Transaction testing

- Test purchase, receipt, delivery, refund and bank reconciliation.
- Verify the customer-facing merchant descriptor.
- Test mobile checkout and Indian/global payment methods available to the account.
- Confirm the post-payment return page.
- Confirm buyer tagging and onboarding emails.

### Days 36–60: Product launch

- Launch the one-time product to newsletter readers and social followers.
- Publish proof-based product walkthroughs.
- Collect product feedback and support questions.
- Record funnel metrics from page visit to checkout to purchase.
- Improve the offer before adding another product.

### Days 61–75: Community validation

- Invite customers to a free beta Discord server or time-boxed paid challenge, depending on Dodo’s approval.
- Use a single paid role and minimal channels.
- Run the Monday, midweek and Friday cadence.
- Measure active members, completed check-ins and repeated requests.

### Days 76–90: Membership decision

- Launch a recurring membership only if the readiness gate is met.
- Create the Dodo subscription and attach the Discord entitlement if approved.
- Test grant, renewal, failed payment, cancellation, refund and revocation using test accounts.
- Otherwise, keep Discord as a time-boxed customer bonus and focus on increasing product sales and newsletter growth.

## Go/no-go checklist

### Website

- GitHub is source control, not commercial Pages hosting.
- Cloudflare Pages is connected and auto-deploying.
- Apex and `www` resolve correctly.
- Privacy, terms, refunds and support pages are public.
- No payment API secret exists in frontend code or repository history.

### Newsletter

- Kit form works on mobile and desktop.
- Lead magnet arrives automatically.
- Branded sending domain is verified.
- Welcome sequence has one clear CTA per email.
- Subscriber consent and unsubscribe flow are tested.

### Dodo

- Written product-category approval is stored.
- Live pages and delivered files match the approved description.
- Individual/sole-proprietor KYC and bank ownership match.
- One-time checkout, refund and file revocation are tested.
- Subscription and Discord entitlements are tested separately.

### Discord

- Dodo bot role sits above the paid-member role.
- Bot has only required permissions.
- Paid channels are invisible to `@everyone`.
- OAuth instructions are clear.
- Failed entitlement and wrong-account support procedures exist.
- Community rules prohibit individualized regulated advice.

## Final recommendation

Proceed with **GitHub repository → Cloudflare Pages → custom domain**, not GitHub Pages. Embed **Kit** forms in the website and use Kit as the free newsletter and audience database. Treat **Discord** as an independent community platform whose paid role can be granted and revoked natively by **Dodo entitlements**.

Do not treat Dodo as confirmed merely because its technology fits. Its current acceptance policy conflicts materially with a combined finance-and-fitness product line. Obtain written approval for the exact downloads, claims and Discord channels first. If approval is granted, Dodo produces a compact, low-maintenance stack. If approval is declined, preserve the same website, newsletter and Discord architecture and replace only checkout and fulfilment with Payhip plus Razorpay or another CA-approved processor.

The correct launch order is therefore: **commercial-safe hosting → newsletter ownership → processor pre-clearance → one useful product → tested checkout → sales → time-boxed Discord challenge → recurring membership → faceless course only after demand is proven.**

---

## References

1. [Merchant Acceptance Policy - Dodo Payments Documentation](https://docs.dodopayments.com/miscellaneous/merchant-acceptance)

2. [GitHub Terms for Additional Products and Features](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features) - Get started, troubleshoot, and make the most of GitHub. Documentation for new users, developers, adm...

3. [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) - Learn about the limits and limitations of GitHub Pages.

4. [Your first deploy](https://developers.cloudflare.com/pages/get-started/git-integration/) - Connect your Git provider to Pages.

5. [Git integration · Cloudflare Pages docs](https://developers.cloudflare.com/pages/configuration/git-integration/) - Connect a GitHub or GitLab repository to Cloudflare Pages for automatic build and deploy on push.

6. [Cloudflare Pages](https://pages.cloudflare.com/?lang=en) - Build your next application with Cloudflare Pages

7. [Custom domains · Cloudflare Pages docs](https://developers.cloudflare.com/pages/configuration/custom-domains/) - Add custom domains and subdomains to your Cloudflare Pages project.

8. [How to use a custom domain for Landing Pages](https://help.kit.com/en/articles/3107877-how-to-use-a-custom-domain-for-landing-pages) - If you want your Landing Page to use your own custom domain rather than ours, here's how to get set ...

9. [The Kit Form builder | Kit Help Center](https://help.kit.com/en/articles/2502640-the-kit-form-builder) - Creating opt-ins to gather subscribers is simple using Kit's Form builder.

10. [Form embedding basics - Kit Help Center](https://help.kit.com/en/articles/4009572-form-embedding-basics) - A basic overview of how to embed Kit Forms on your blog or website.

11. [Verify your domain to optimize your deliverability | Kit Help ...](https://help.kit.com/en/articles/2502558-verify-your-domain-to-optimize-your-deliverability) - Advanced custom options for your deliverability.

12. [What you should know before setting up a Verified Sending Domain](https://help.kit.com/en/articles/9176509-what-you-should-know-before-setting-up-a-verified-sending-domain)

13. [Introduction - Dodo Payments Documentation](https://docs.dodopayments.com/features/mor-introduction)

14. [Dodo Payments Pricing — 0% Setup Fee, Pay Only When You ...](https://dodopayments.com/pricing)

15. [One-time Payments Integration Guide](https://docs.dodopayments.com/id/developer-resources/integration-guide) - 1. Checkout Sessions · 2. Overlay Checkout · 3. Inline Checkout · 4. Static Payment Links · 4. Dynam...

16. [GoHighLevel](https://docs.dodopayments.com/id/integrations/gohighlevel)

17. [FAQs - Dodo Payments Documentation](https://docs.dodopayments.com/miscellaneous/faq)

18. [Discord Entitlement - Dodo Payments Documentation](https://docs.dodopayments.com/features/entitlements/discord) - Grant your customers a role in your Discord server when they purchase, and revoke it automatically w...

19. [llms-full.txt](https://docs.dodopayments.com/llms-full.txt)

20. [v1.97.6 (May 7, 2026) - Dodo Payments Documentation](https://docs.dodopayments.com/changelog/v1.97.6)

21. [Entitlement Grant - Dodo Payments Documentation](https://docs.dodopayments.com/developer-resources/webhooks/intents/entitlement-grant) - These events fire whenever a customer's entitlement grant changes state, for example when a license ...

22. [Permissions on Discord](https://discord.com/community/permissions-on-discord-discord) - Once you become a moderator, it’s important to know what tools are at your disposal to help manage y...

23. [Discord Roles and Permissions](https://support.discord.com/hc/en-us/articles/214836687-Discord-Roles-and-Permissions) - Learn the fundamentals of Discord roles and permissions—the key tools for running an organized and s...

24. [How Do I Create a Community For My Game? - Documentation](https://docs.discord.com/developers/game-development/how-to-create-a-community-for-your-game)

25. [Introduction - Dodo Payments Documentation](https://docs.dodopayments.com/features/entitlements/introduction) - An entitlement is a reusable definition of something you deliver to a customer: a Pro license key, a...

26. [Review & Monitoring Policy - Dodo Payments Documentation](https://docs.dodopayments.com/miscellaneous/review-monitoring-policy)

27. [What's New at Payhip: 2025 Feature Round‑Up](https://payhip.com/blog/whats-new-at-payhip-2025/) - You asked, we listened, and we got to work. Behind the scenes, we have been pushing harder than ever...

28. [Connect Your Razorpay Account](https://help.payhip.com/article/368-connect-your-razorpay-account) - In your Payhip dashboard, go to Account → Settings → Payment Details. You'll see a list of payment g...

29. [FAQ - Payhip](https://payhip.com/faq) - These are the most common questions we get asked when selling digital downloads and memberships on P...

30. [Payhip Integrations | Connect Your Apps with Zapier](https://zapier.com/apps/payhip/integrations) - Instantly connect Payhip with the apps you use everyday. Payhip integrates with 9,000 other apps on ...

31. [Payhip Discord Integration - Quick Connect](https://zapier.com/apps/payhip/integrations/discord) - Integrate Payhip and Discord in a few minutes. Quickly connect Payhip and Discord with over 9000 app...

32. [The Kit Newsletter Plan - Kit Help Center](https://help.kit.com/en/articles/9053602-the-kit-newsletter-plan)

33. [The Kit Free plan | Kit Help Center](https://help.kit.com/en/articles/16627071-the-kit-free-plan) - Kit's Free plan gives you up to 10,000 email subscribers, unlimited forms and landing pages, and unl...

