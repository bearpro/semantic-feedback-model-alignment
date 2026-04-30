# GitHub REST API — Create an issue normalized fragment

The API operation creates a new issue in a GitHub repository.

The HTTP method is POST. The request path is `/repos/{owner}/{repo}/issues`. The `owner` path parameter is required and identifies the account owner of the repository. The owner name is not case sensitive. The `repo` path parameter is required and identifies the repository name without the `.git` extension. The repository name is not case sensitive.

Any user with pull access to a repository can create an issue. If issues are disabled in the repository, the API returns status code 410 Gone. Creating an issue triggers notifications. Creating content too quickly can result in secondary rate limiting.

Fine-grained access can be provided by a GitHub App user access token, a GitHub App installation access token, or a fine-grained personal access token. The token must have write permission for repository Issues.

The recommended `accept` request header is `application/vnd.github+json`. The endpoint also supports media types that determine whether the response contains the raw Markdown issue body, a plain-text issue body, an HTML-rendered issue body, or all three body representations.

The request body has the following fields:

- `title`: required. The title of the issue. The value can be a string or an integer.
- `body`: optional. The contents of the issue as a string.
- `milestone`: optional. Null, string, or integer. It identifies the milestone number to associate with the new issue.
- `labels`: optional. Array. Labels to associate with the issue.
- `assignees`: optional. Array of strings. Each string is a user login to assign to the issue.
- `type`: optional. String or null. It is the name of the issue type to associate with the issue.

Only users with push access can set the milestone, labels, assignees, or type on a new issue. If a caller lacks push access, those fields can be silently dropped.

The operation can return the following HTTP status codes: 201 Created, 400 Bad Request, 403 Forbidden, 404 Resource Not Found, 410 Gone, 422 Validation Failed, and 503 Service Unavailable.

A created issue resource contains identifiers, URLs, state information, textual content, people, labels, and related issue metadata. Typical response fields include `id`, `node_id`, `url`, `repository_url`, `labels_url`, `comments_url`, `events_url`, `html_url`, `number`, `state`, `title`, `body`, `user`, `labels`, `assignees`, `milestone`, `locked`, `active_lock_reason`, `comments`, `pull_request`, `closed_at`, `created_at`, `updated_at`, `author_association`, and `state_reason`.

A label object can contain an identifier, URL, name, description, color, and default flag. A milestone object can contain a URL, HTML URL, labels URL, identifier, number, state, title, description, creator, count of open issues, count of closed issues, creation time, update time, closing time, and due date. A user object can contain login, identifier, profile URLs, avatar URL, user type, and site-admin flag.
