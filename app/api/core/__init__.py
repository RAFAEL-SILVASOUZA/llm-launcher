"""Core do launcher — portado de models.py, sem UI/CLI.

Cada módulo é independente da camada HTTP (server.py): só recebe/retorna
dados puros (dict, dataclass, Path). Reusa toda a lógica de:
  - parsing de metadados GGUF
  - estimador de VRAM/RAM
  - builder de comando llama-server/llama-cli
  - classificador de falhas + escada de auto-degrade
  - persistência de configurações
"""
