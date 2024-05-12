from opmentis import register_miners

miner_wallet_address = "miner_wallet_address"
miner = register_miners(wallet_address=miner_wallet_address)
print("Miner Registered:", miner)