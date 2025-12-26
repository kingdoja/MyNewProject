#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinIO 上传工具封装

简化使用方式：
    uploader = MinioUploader(
        endpoint="127.0.0.1:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="rtdetr",
        secure=False,
        base_path="rtdetr/results"
    )
    uploader.upload_file("/path/to/file.json", object_name="sample.json")
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

try:
    from minio import Minio
    from minio.error import S3Error  # type: ignore
except Exception as exc:  # pragma: no cover - 运行时依赖
    raise ImportError(
        "缺少 minio 依赖，请先安装：pip install minio"
    ) from exc


class MinioUploader:
    """MinIO 上传封装"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        base_path: str | None = None,
        region: Optional[str] = None,
        enable_bucket_create: bool = True,
    ) -> None:
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.bucket = bucket
        self.base_path = base_path.strip("/") if base_path else ""
        self.enable_bucket_create = enable_bucket_create

        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """确保 bucket 存在，不存在则创建"""
        exists = self.client.bucket_exists(self.bucket)
        if not exists:
            if not self.enable_bucket_create:
                raise RuntimeError(f"MinIO bucket 不存在且未允许创建: {self.bucket}")
            self.client.make_bucket(self.bucket)

    def _full_object_name(self, object_name: str) -> str:
        object_name = object_name.lstrip("/")
        if self.base_path:
            return f"{self.base_path}/{object_name}"
        return object_name

    def upload_file(self, local_path: Path, object_name: Optional[str] = None) -> str:
        """上传单个文件

        Args:
            local_path: 本地文件路径
            object_name: MinIO 对象名（可选，默认使用文件名并带上 base_path）

        Returns:
            已上传对象的完整路径（bucket/object）
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"待上传文件不存在: {path}")

        object_name = object_name or path.name
        object_name = self._full_object_name(object_name)

        content_type, _ = mimetypes.guess_type(str(path))

        self.client.fput_object(
            bucket_name=self.bucket,
            object_name=object_name,
            file_path=str(path),
            content_type=content_type or "application/octet-stream",
        )
        return f"{self.bucket}/{object_name}"


