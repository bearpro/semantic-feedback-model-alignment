# Stripe API — PaymentIntent normalized fragment

A PaymentIntent represents Stripe's intent to collect a payment from a customer. A system should normally create one PaymentIntent for each order or customer checkout session. The PaymentIntent tracks the payment attempt across creation, confirmation, possible customer authentication, processing, capture, cancellation, and success.

The API operation `Create a PaymentIntent` creates a PaymentIntent object. After creation, a payment method can be attached and the PaymentIntent can be confirmed. If the create request includes `confirm=true`, creation and confirmation are attempted in the same operation.

The create request has two required parameters. `amount` is an integer amount intended to be collected. It is expressed in the smallest currency unit, such as cents for USD. The value must be positive, must meet Stripe's minimum charge rules, and supports up to eight digits. `currency` is a required lowercase three-letter ISO currency code, and it must be one of Stripe's supported currencies.

The create request can include optional fields. `automatic_payment_methods` is an object that can enable payment methods configured in the Stripe Dashboard and compatible with the PaymentIntent. `automatic_payment_methods.enabled` is a boolean. `automatic_payment_methods.allow_redirects` controls whether redirect-based methods are allowed; values are `always` or `never`.

`capture_method` controls when funds are captured after authorization. Its values are `automatic`, `automatic_async`, and `manual`. With automatic capture, funds are captured when the customer authorizes the payment. With automatic asynchronous capture, Stripe captures asynchronously after authorization. With manual capture, funds are placed on hold and must be captured later.

`confirm` is an optional boolean that defaults to false. When it is true, the PaymentIntent is confirmed immediately during creation. `confirmation_method` describes how confirmation is performed. Its values are `automatic` and `manual`. With automatic confirmation, the PaymentIntent can be confirmed using a publishable key. With manual confirmation, payment attempts must be made from a server using a secret key.

Other optional request fields include `customer`, `description`, `metadata`, `receipt_email`, `payment_method`, `payment_method_data`, `payment_method_types`, `setup_future_usage`, `shipping`, `statement_descriptor`, `statement_descriptor_suffix`, `transfer_data`, and `transfer_group`.

A PaymentIntent object contains the following important attributes:

- `id`: unique identifier of the PaymentIntent.
- `object`: object type string.
- `amount`: intended amount to collect in the smallest currency unit.
- `amount_capturable`: amount that can currently be captured.
- `amount_received`: amount received so far.
- `currency`: lowercase three-letter currency code.
- `capture_method`: method used for capturing funds.
- `confirmation_method`: method used for confirming the payment.
- `created`: Unix timestamp for object creation.
- `customer`: customer identifier, if a customer is attached.
- `description`: optional description.
- `livemode`: boolean indicating whether the object exists in live mode.
- `metadata`: key-value pairs attached to the object.
- `payment_method`: attached payment method identifier, if present.
- `payment_method_types`: array of payment method type names.
- `latest_charge`: identifier of the latest Charge object, or null before a confirmation attempt creates one.
- `next_action`: object describing customer or integration actions required to complete the payment, or null.
- `processing`: object describing processing state, or null.
- `receipt_email`: email address for receipt delivery, if provided.
- `canceled_at`: cancellation timestamp, or null.
- `cancellation_reason`: reason for cancellation, or null.
- `status`: lifecycle status of the PaymentIntent.

The `status` enum can have the following values: `requires_payment_method`, `requires_confirmation`, `requires_action`, `processing`, `requires_capture`, `canceled`, and `succeeded`. `requires_payment_method` means a payment method still needs to be attached. `requires_confirmation` means the PaymentIntent requires confirmation. `requires_action` means the customer must perform an additional action. `processing` means the payment is being processed. `requires_capture` means the payment was confirmed and must be captured. `canceled` means the PaymentIntent was canceled. `succeeded` means the payment succeeded.
