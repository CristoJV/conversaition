#!/usr/bin/env python3

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------

# Formato esperado:
#
# 12/09/2026, 21:37 - Candela: Hola
#
# También soporta segundos:
#
# 12/09/2026, 21:37:15 - Candela: Hola

MESSAGE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), "
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?) - "
)


@dataclass(frozen=True)
class Message:
    timestamp: datetime
    text: str
    source: Path

    @property
    def signature(self) -> str:
        """
        Representación usada para comparar mensajes entre backups.

        Importante:
        no se deduplican mensajes individualmente.

        La firma se usa para encontrar secuencias comunes entre dos
        backups, por lo que dos mensajes idénticos consecutivos siguen
        siendo dos mensajes distintos dentro de la secuencia.
        """
        return normalize_message(self.text)


def normalize_message(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def parse_timestamp(line: str) -> datetime:
    match = MESSAGE_RE.match(line)

    if not match:
        raise ValueError(f"No se puede obtener la fecha de:\n{line}")

    value = f"{match.group('date')} {match.group('time')}"

    formats = (
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%y %H:%M:%S",
    )

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(f"Formato de fecha no reconocido: {value}")


def looks_like_whatsapp_chat(path: Path) -> bool:
    """
    Evita considerar cualquier .txt de la carpeta como un chat.
    """

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
        ) as f:
            # Normalmente la primera línea ya sirve,
            # pero miramos unas cuantas por seguridad.
            for _ in range(100):
                line = f.readline()

                if not line:
                    break

                if MESSAGE_RE.match(line):
                    return True

    except OSError:
        return False

    return False


def read_chat(path: Path) -> list[Message]:
    """
    Lee un export de WhatsApp conservando mensajes multilínea.
    """

    messages: list[Message] = []
    current: list[str] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")

            if MESSAGE_RE.match(line):
                if current:
                    text = "\n".join(current)

                    messages.append(
                        Message(
                            timestamp=parse_timestamp(current[0]),
                            text=text,
                            source=path,
                        )
                    )

                current = [line]

            elif current:
                # Continuación del mensaje anterior.
                current.append(line)

        if current:
            text = "\n".join(current)

            messages.append(
                Message(
                    timestamp=parse_timestamp(current[0]),
                    text=text,
                    source=path,
                )
            )

    return messages


# ---------------------------------------------------------------------
# Búsqueda eficiente de secuencias
# ---------------------------------------------------------------------


def prefix_function(pattern: list[str]) -> list[int]:
    """
    Prefix function de KMP.
    """

    pi = [0] * len(pattern)
    j = 0

    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]:
            j = pi[j - 1]

        if pattern[i] == pattern[j]:
            j += 1

        pi[i] = j

    return pi


def find_subsequence(
    sequence: list[str],
    pattern: list[str],
) -> int | None:
    """
    Devuelve el índice donde empieza pattern dentro de sequence.

    Usa KMP, por lo que no tenemos que hacer una búsqueda cuadrática.
    """

    if not pattern:
        return 0

    pi = prefix_function(pattern)
    j = 0

    for i, value in enumerate(sequence):
        while j and value != pattern[j]:
            j = pi[j - 1]

        if value == pattern[j]:
            j += 1

            if j == len(pattern):
                return i - len(pattern) + 1

    return None


def suffix_prefix_overlap(
    left: list[str],
    right: list[str],
) -> int:
    """
    Encuentra el mayor K tal que:

        últimos K mensajes de left
        ==
        primeros K mensajes de right

    Ejemplo:

        left:
            A B C D E

        right:
                  D E F G

        resultado: 2
    """

    if not left or not right:
        return 0

    pi = prefix_function(right)
    j = 0

    for i, value in enumerate(left):
        while j and value != right[j]:
            j = pi[j - 1]

        if value == right[j]:
            j += 1

            if j == len(right):
                # Si hemos terminado justo al final de left,
                # right entero es el sufijo.
                if i == len(left) - 1:
                    return j

                # Si quedan elementos en left, continuamos buscando
                # un posible sufijo más adelante.
                j = pi[j - 1]

    return j


# ---------------------------------------------------------------------
# Merge de chats
# ---------------------------------------------------------------------


def merge_two_chats(
    left: list[Message],
    right: list[Message],
    min_overlap: int,
) -> tuple[list[Message], dict[str, Any]]:
    if not left:
        return right, {
            "type": "initial",
            "added": len(right),
        }

    if not right:
        return left, {
            "type": "empty",
            "added": 0,
        }

    left_signatures = [message.signature for message in left]

    right_signatures = [message.signature for message in right]

    # -------------------------------------------------------------
    # ¿Está right completamente contenido en left?
    # -------------------------------------------------------------

    right_inside_left = find_subsequence(
        left_signatures,
        right_signatures,
    )

    if right_inside_left is not None:
        return left, {
            "type": "contained",
            "contained": "right",
            "overlap": len(right),
            "added": 0,
        }

    # -------------------------------------------------------------
    # ¿Está left completamente contenido en right?
    # -------------------------------------------------------------

    left_inside_right = find_subsequence(
        right_signatures,
        left_signatures,
    )

    if left_inside_right is not None:
        return right, {
            "type": "contained",
            "contained": "left",
            "overlap": len(left),
            "added": len(right) - len(left),
        }

    # -------------------------------------------------------------
    # Solapamiento normal:
    #
    # left:   A B C D E
    # right:        D E F G
    # -------------------------------------------------------------

    overlap = suffix_prefix_overlap(
        left_signatures,
        right_signatures,
    )

    if overlap >= min_overlap:
        merged = left + right[overlap:]

        return merged, {
            "type": "overlap",
            "overlap": overlap,
            "added": len(right) - overlap,
        }

    # -------------------------------------------------------------
    # Por seguridad comprobamos también el sentido inverso.
    # -------------------------------------------------------------

    reverse_overlap = suffix_prefix_overlap(
        right_signatures,
        left_signatures,
    )

    if reverse_overlap >= min_overlap:
        merged = right + left[reverse_overlap:]

        return merged, {
            "type": "reverse-overlap",
            "overlap": reverse_overlap,
            "added": len(left) - reverse_overlap,
        }

    # -------------------------------------------------------------
    # No hay solapamiento, pero los chats son temporalmente
    # completamente disjuntos.
    #
    # En ese caso concatenarlos es seguro.
    # -------------------------------------------------------------

    if left[-1].timestamp < right[0].timestamp:
        return left + right, {
            "type": "gap",
            "overlap": 0,
            "added": len(right),
        }

    if right[-1].timestamp < left[0].timestamp:
        return right + left, {
            "type": "reverse-gap",
            "overlap": 0,
            "added": len(left),
        }

    # -------------------------------------------------------------
    # Los rangos temporales se pisan, pero no hemos encontrado una
    # secuencia de mensajes suficientemente fiable.
    #
    # Aquí NO intentamos adivinar.
    # -------------------------------------------------------------

    raise RuntimeError(
        "\nNo puedo fusionar estos dos backups de forma segura.\n"
        "\n"
        f"Chat acumulado:\n"
        f"  {left[0].timestamp} -> {left[-1].timestamp}\n"
        "\n"
        f"Nuevo backup:\n"
        f"  {right[0].source}\n"
        f"  {right[0].timestamp} -> {right[-1].timestamp}\n"
        "\n"
        f"Mayor solapamiento encontrado: {overlap} mensajes\n"
        f"Solapamiento inverso: {reverse_overlap} mensajes\n"
        f"Mínimo requerido: {min_overlap}\n"
        "\n"
        "Los rangos de fechas se solapan pero no existe una secuencia "
        "común suficientemente larga. No voy a eliminar mensajes "
        "automáticamente porque podría perder información.\n"
    )


def merge_all_chats(
    chats: dict[Path, list[Message]],
    min_overlap: int,
) -> tuple[list[Message], list[dict[str, Any]]]:
    if not chats:
        return [], []

    # Ordenamos por:
    #
    # 1. primer mensaje
    # 2. último mensaje
    #
    # No dependemos del nombre de carpeta ni de archivo.

    ordered = sorted(
        chats.items(),
        key=lambda item: (
            item[1][0].timestamp,
            item[1][-1].timestamp,
        ),
    )

    merged: list[Message] = []
    report: list[dict[str, Any]] = []

    for path, messages in ordered:
        before = len(merged)

        merged, info = merge_two_chats(
            merged,
            messages,
            min_overlap=min_overlap,
        )

        info.update(
            {
                "source": str(path),
                "source_messages": len(messages),
                "messages_before": before,
                "messages_after": len(merged),
            }
        )

        report.append(info)

    return merged, report


def write_chat(
    messages: list[Message],
    output: Path,
) -> None:
    with output.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for message in messages:
            f.write(message.text)
            f.write("\n")


# ---------------------------------------------------------------------
# Descubrimiento automático de backups
# ---------------------------------------------------------------------


def is_inside(
    path: Path,
    directory: Path,
) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def find_chats(
    backups_dir: Path,
    output_dir: Path,
) -> list[Path]:
    if not backups_dir.exists():
        raise FileNotFoundError(f"No existe: {backups_dir}")

    if not backups_dir.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {backups_dir}")

    backups_resolved = backups_dir.resolve()
    output_resolved = output_dir.resolve()

    # Solo necesitamos excluir output si output está DENTRO
    # de la carpeta de backups.
    output_inside_backups = is_inside(
        output_resolved,
        backups_resolved,
    )

    candidates: list[Path] = []

    for path in backups_dir.rglob("*"):
        if not path.is_file():
            continue

        # Ignorar nuestro propio output únicamente cuando output
        # está realmente dentro del árbol de backups.
        if output_inside_backups and is_inside(path, output_resolved):
            continue

        # Linux distingue .txt de .TXT, así que comprobamos
        # la extensión de forma case-insensitive.
        if path.suffix.lower() != ".txt":
            continue

        if looks_like_whatsapp_chat(path):
            candidates.append(path)

    candidates.sort()

    return candidates


# ---------------------------------------------------------------------
# Archivos multimedia
# ---------------------------------------------------------------------


@dataclass
class StoredFile:
    path: Path
    size: int
    digest: bytes


def file_hash(path: Path) -> bytes:
    """
    Hash rápido para localizar posibles duplicados.

    La decisión definitiva NO depende solamente del hash:
    después usamos filecmp con shallow=False.
    """

    hasher = hashlib.blake2b(
        digest_size=32,
    )

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.digest()


class MediaStore:
    def __init__(
        self,
        output: Path,
    ):
        self.output = output

        self.output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # (tamaño, hash) -> archivo canónico
        self.by_content: dict[
            tuple[int, bytes],
            StoredFile,
        ] = {}

        self.stats = {
            "input_files": 0,
            "copied": 0,
            "duplicates": 0,
            "hardlinks": 0,
            "existing": 0,
        }

        self.duplicates: list[dict[str, str]] = []

    def add(
        self,
        source: Path,
    ) -> Path:
        self.stats["input_files"] += 1

        size = source.stat().st_size
        digest = file_hash(source)

        key = (
            size,
            digest,
        )

        canonical = self.by_content.get(key)

        # ---------------------------------------------------------
        # Contenido posiblemente duplicado.
        #
        # El hash solamente localiza el candidato.
        # filecmp comprueba los bytes reales.
        # ---------------------------------------------------------

        if canonical is not None:
            if filecmp.cmp(
                source,
                canonical.path,
                shallow=False,
            ):
                self.stats["duplicates"] += 1

                self.duplicates.append(
                    {
                        "source": str(source),
                        "canonical": str(canonical.path),
                    }
                )

                return self.create_alias(
                    canonical=canonical.path,
                    name=source.name,
                )

        destination = self.output / source.name

        # ---------------------------------------------------------
        # Ya existe exactamente ese nombre.
        # ---------------------------------------------------------

        if destination.exists():
            if filecmp.cmp(
                source,
                destination,
                shallow=False,
            ):
                self.stats["existing"] += 1

                self.by_content[key] = StoredFile(
                    path=destination,
                    size=size,
                    digest=digest,
                )

                print(f"  SAME      {source.name}")

                return destination

            # No lo renombramos automáticamente.
            #
            # El chat referencia al adjunto por nombre, así que cambiar
            # el nombre sin modificar también el chat podría romper la
            # relación.
            raise RuntimeError(
                "\nConflicto de archivos multimedia.\n"
                "\n"
                "Dos archivos tienen exactamente el mismo nombre "
                "pero distinto contenido:\n"
                "\n"
                f"  {source}\n"
                f"  {destination}\n"
                "\n"
                "No voy a sobreescribir ni renombrar automáticamente "
                "el archivo porque el chat puede referenciarlo por "
                "su nombre.\n"
            )

        # ---------------------------------------------------------
        # Archivo nuevo.
        # ---------------------------------------------------------

        shutil.copy2(
            source,
            destination,
        )

        self.by_content[key] = StoredFile(
            path=destination,
            size=size,
            digest=digest,
        )

        self.stats["copied"] += 1

        print(f"  COPY      {source.name}")

        return destination

    def create_alias(
        self,
        canonical: Path,
        name: str,
    ) -> Path:
        destination = self.output / name

        # Mismo nombre.
        if destination == canonical:
            return canonical

        if destination.exists():
            if filecmp.cmp(
                canonical,
                destination,
                shallow=False,
            ):
                self.stats["existing"] += 1

                print(f"  SAME      {name}")

                return destination

            raise RuntimeError(
                f"\nConflicto de nombre en archivos duplicados:\n  {destination}\n"
            )

        # ---------------------------------------------------------
        # Hardlink:
        #
        # dos nombres diferentes,
        # mismos bytes físicos en disco.
        # ---------------------------------------------------------

        try:
            os.link(
                canonical,
                destination,
            )

            self.stats["hardlinks"] += 1

            print(f"  LINK      {name} -> {canonical.name}")

        except OSError:
            # Por ejemplo:
            # - filesystems distintos
            # - filesystem sin hardlinks
            #
            # En ese caso simplemente copiamos.
            shutil.copy2(
                canonical,
                destination,
            )

            self.stats["copied"] += 1

            print(f"  COPY DUP  {name} -> {canonical.name}")

        return destination


def find_media_files(
    chat_paths: list[Path],
) -> list[Path]:
    """
    Consideramos adjuntos los archivos que están junto al .txt
    de cada export.

    No recorremos recursivamente cada carpeta individual porque
    backups_dir ya es quien representa la colección de exports.
    """

    chat_set = {path.resolve() for path in chat_paths}

    media: set[Path] = set()

    for chat_path in chat_paths:
        directory = chat_path.parent

        for path in directory.iterdir():
            if not path.is_file():
                continue

            resolved = path.resolve()

            if resolved in chat_set:
                continue

            # Los .txt adicionales tampoco se consideran multimedia.
            if path.suffix.lower() == ".txt":
                continue

            media.add(path)

    return sorted(
        media,
        key=lambda path: str(path),
    )


def merge_media(
    chat_paths: list[Path],
    output: Path,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    store = MediaStore(output)

    media_files = find_media_files(
        chat_paths,
    )

    for path in media_files:
        store.add(path)

    return (
        store.stats,
        store.duplicates,
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def write_report(
    path: Path,
    *,
    backups_dir: Path,
    chat_paths: list[Path],
    chats: dict[Path, list[Message]],
    merged: list[Message],
    merge_report: list[dict[str, Any]],
    media_stats: dict[str, int],
    media_duplicates: list[dict[str, str]],
) -> None:
    total_input_messages = sum(len(messages) for messages in chats.values())

    report = {
        "backups_directory": str(backups_dir.resolve()),
        "chat_files": [str(path.resolve()) for path in chat_paths],
        "chat_count": len(chat_paths),
        "messages": {
            "input": total_input_messages,
            "output": len(merged),
            "removed_by_overlap": (total_input_messages - len(merged)),
        },
        "merges": merge_report,
        "media": media_stats,
        "media_duplicates": media_duplicates,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Fusiona varias exportaciones solapadas de un chat de WhatsApp.")
    )

    parser.add_argument(
        "backups",
        type=Path,
        help=(
            "Carpeta que contiene todas las exportaciones del mismo chat de WhatsApp."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=("Carpeta donde escribir el chat y los adjuntos fusionados."),
    )

    parser.add_argument(
        "--min-overlap",
        type=int,
        default=3,
        help=(
            "Número mínimo de mensajes consecutivos idénticos "
            "para aceptar automáticamente un solapamiento "
            "(default: 3)."
        ),
    )

    args = parser.parse_args()

    backups_dir: Path = args.backups
    output_dir: Path = args.output
    min_overlap: int = args.min_overlap

    if min_overlap < 1:
        parser.error("--min-overlap debe ser >= 1")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Buscar chats
    # -------------------------------------------------------------

    print()
    print("== Buscando backups ==")

    chat_paths = find_chats(
        backups_dir,
        output_dir,
    )

    if not chat_paths:
        raise RuntimeError(
            f"No he encontrado ningún export de WhatsApp en:\n  {backups_dir}"
        )

    for path in chat_paths:
        print(f"  {path}")

    print()
    print(f"Backups encontrados: {len(chat_paths)}")

    # -------------------------------------------------------------
    # Leer chats
    # -------------------------------------------------------------

    print()
    print("== Leyendo chats ==")

    chats: dict[
        Path,
        list[Message],
    ] = {}

    for path in chat_paths:
        messages = read_chat(path)

        if not messages:
            continue

        chats[path] = messages

        print(
            f"  {path}: "
            f"{len(messages)} mensajes "
            f"({messages[0].timestamp} -> "
            f"{messages[-1].timestamp})"
        )

    if not chats:
        raise RuntimeError("Los archivos encontrados no contienen mensajes.")

    # -------------------------------------------------------------
    # Fusionar chats
    # -------------------------------------------------------------

    print()
    print("== Fusionando chats ==")

    merged, merge_report = merge_all_chats(
        chats,
        min_overlap=min_overlap,
    )

    for info in merge_report:
        source = Path(info["source"]).name
        merge_type = info["type"]

        print(
            f"  {source}: "
            f"{merge_type}, "
            f"{info.get('overlap', 0)} solapados, "
            f"{info['added']} añadidos"
        )

    chat_output = output_dir / "chat.txt"

    write_chat(
        merged,
        chat_output,
    )

    # -------------------------------------------------------------
    # Fusionar multimedia
    # -------------------------------------------------------------

    print()
    print("== Fusionando multimedia ==")

    media_stats, media_duplicates = merge_media(
        list(chats.keys()),
        output_dir / "media",
    )

    # -------------------------------------------------------------
    # Informe
    # -------------------------------------------------------------

    report_output = output_dir / "merge-report.json"

    write_report(
        report_output,
        backups_dir=backups_dir,
        chat_paths=chat_paths,
        chats=chats,
        merged=merged,
        merge_report=merge_report,
        media_stats=media_stats,
        media_duplicates=media_duplicates,
    )

    # -------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------

    total_input_messages = sum(len(messages) for messages in chats.values())

    print()
    print("== Resultado ==")
    print()
    print(f"  Backups:             {len(chats)}")
    print(f"  Mensajes de entrada: {total_input_messages}")
    print(f"  Mensajes finales:    {len(merged)}")
    print(f"  Multimedia entrada:  {media_stats['input_files']}")
    print(f"  Multimedia copiada:  {media_stats['copied']}")
    print(f"  Duplicados:          {media_stats['duplicates']}")
    print(f"  Hardlinks creados:   {media_stats['hardlinks']}")

    print()
    print(f"  Chat:    {chat_output}")
    print(f"  Media:   {output_dir / 'media'}")
    print(f"  Informe: {report_output}")
    print()


if __name__ == "__main__":
    main()
