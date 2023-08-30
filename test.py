"""
A function to retrieve a secret from an Azure Keyvauklt instance using a user assigned managed identity.
"""
def get_secret_from_keyvault(secret_name: str, keyvault_name: str, keyvault_uri: str) -> str:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=keyvault_uri, credential=credential)
    secret = client.get_secret(secret_name)
    return secret.value

