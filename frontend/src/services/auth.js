const config = {
  domain: import.meta.env.VITE_COGNITO_DOMAIN,
  clientId: import.meta.env.VITE_COGNITO_CLIENT_ID,
  redirectUri: import.meta.env.VITE_COGNITO_REDIRECT_URI || window.location.origin,
};

let accessToken = null;

export const cognitoConfigured = Boolean(config.domain && config.clientId);
export const currentAccessToken = () => accessToken;

export async function startSignIn() {
  if (!cognitoConfigured) throw new Error("Cognito is not configured for this frontend.");
  const verifier = randomString(64);
  const state = randomString(32);
  sessionStorage.setItem("fincloud_pkce_verifier", verifier);
  sessionStorage.setItem("fincloud_oauth_state", state);
  const challenge = await sha256Base64Url(verifier);
  const url = new URL("/oauth2/authorize", normalizeDomain(config.domain));
  url.search = new URLSearchParams({
    client_id: config.clientId,
    response_type: "code",
    scope: "openid email",
    redirect_uri: config.redirectUri,
    state,
    code_challenge_method: "S256",
    code_challenge: challenge,
  }).toString();
  window.location.assign(url);
}

export async function finishSignIn() {
  const current = new URL(window.location.href);
  const code = current.searchParams.get("code");
  const providerError = current.searchParams.get("error");
  if (providerError) {
    const description = current.searchParams.get("error_description");
    throw new Error(description || "Cognito sign-in was not completed.");
  }
  if (!code) return false;
  const state = current.searchParams.get("state");
  const expectedState = sessionStorage.getItem("fincloud_oauth_state");
  const verifier = sessionStorage.getItem("fincloud_pkce_verifier");
  sessionStorage.removeItem("fincloud_oauth_state");
  sessionStorage.removeItem("fincloud_pkce_verifier");
  current.searchParams.delete("code");
  current.searchParams.delete("state");
  current.searchParams.delete("error");
  current.searchParams.delete("error_description");
  window.history.replaceState({}, document.title, current.pathname + current.search + current.hash);
  if (!state || state !== expectedState || !verifier) {
    throw new Error("Cognito sign-in state validation failed. Please try again.");
  }
  const response = await fetch(`${normalizeDomain(config.domain)}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: config.clientId,
      code,
      redirect_uri: config.redirectUri,
      code_verifier: verifier,
    }),
  });
  const payload = await response.json();
  if (!response.ok || typeof payload.access_token !== "string") {
    throw new Error("Cognito could not exchange the authorization code.");
  }
  accessToken = payload.access_token;
  return true;
}

export function signOut() {
  accessToken = null;
  if (!cognitoConfigured) return;
  const url = new URL("/logout", normalizeDomain(config.domain));
  url.search = new URLSearchParams({
    client_id: config.clientId,
    logout_uri: config.redirectUri,
  }).toString();
  window.location.assign(url);
}

function normalizeDomain(domain) {
  return domain.startsWith("http") ? domain : `https://${domain}`;
}

function randomString(length) {
  const bytes = crypto.getRandomValues(new Uint8Array(length));
  return base64Url(bytes);
}

async function sha256Base64Url(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return base64Url(new Uint8Array(digest));
}

function base64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
