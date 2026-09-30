# lru_cache：Python 标准库 functools 内置的 LRU（Least Recently Used，最近最少使用）缓存装饰器。
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

"""
它利用了 pydantic-settings 模块从本地 .env 文件或系统环境变量中加载配置，
并通过 functools.lru_cache 实现了配置对象的单例缓存。
在 Agent 应用开发中，我们通常需要管理很多配置项（例如：OPENAI_API_KEY、数据库连接串、模型名称、调试开关等）。

这段代码的作用是：定义并全局管理应用配置。
它会自动读取环境变量/.env文件，验证数据类型，并保证全局只初始化一次配置对象，避免重复读取文件的开销。
"""
class Settings(BaseSettings):
    # 如果你在 .env 文件里定义了 app_name 和 debug 两个变量，Pydantic 会自动覆盖代码中的默认值。
    # 例如：
    # APP_NAME="My Premium Agent"
    # DEBUG=False
    # Pydantic 会自动覆盖代码中的默认值，把 app_name 变成 "My Premium Agent"，
    # 把 debug 转换成布尔值 False（Pydantic 会自动做类型转换，比如把字符串 "false" 转为 False）。
    app_name: str = "SmartMall Agent"
    app_version: str = "0.1.0"
    debug: bool = True

    # 用来控制pydantic如何读取环境变量
    model_config = SettingsConfigDict(
        env_file=".env",# 告诉取根目录找.env文件
        env_file_encoding="utf-8",# 读取env使用的编码
        extra="ignore",# 如果env文件里定义了额外的变量，但是settings里面没有定义则直接忽略，不会报错
    )

# @lru_cache 装饰器：
# 当你第一次调用 get_settings() 时，它会执行 Settings() 去读取 .env 文件并解析配置。
# 当你第 2 次及以后调用 get_settings() 时，它会直接返回第一次创建好的内存对象，不再重新去读取磁盘上的 .env 文件。
# 磁盘 IO 是很慢的。
# 在 Agent 运行过程中，可能每处理一个用户 Prompt、每调用一次工具都要用到 API Key 或配置，
# 如果每次都读取文件会严重拖慢性能。lru_cache 实现了全局单例模式（Singleton）。
@lru_cache
def get_settings() -> Settings:
    return Settings()