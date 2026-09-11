from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from hierarchical.postprocessor import ResultPostprocessor
from hierarchy_annotator_backend_v4_inlocal import HierarchyAnnotator
from doc_processor import DocProcessor
import os, json
import torch
# import pdfminer_header_hierarcy_detection
# from pdfminer_header_hierarcy_detection import extract_lines_with_style
# from docling_json_pre_processor import enrich_docling_json, attach_pdfminer_styles_to_docling
# from yolo_header_dectection import main
# from yolo_json_pre_processing import promote_headers_from_yolo
# import llm_header_preprocessing
from header_hierarchy_post_processor import process_h_columns
from fragment_post_processor import fragment_main


# -------------------------------------------------
# DEVICE CONFIGURATION (CUDA -> CPU fallback)
# -------------------------------------------------
# if torch.cuda.is_available():
#     device = "cuda"
#     print("Using CUDA for processing")
# elif device == "mps":
#     print("Using MPS for processing")
# else:
#     device = "cpu"
#     print("CUDA not available. Falling back to CPU")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device} for processing")


# --- input/output Configuration ---
# folder_name = "pdfs"
# base_path = "reg_test/new_regs/common_issues"
folder_name = "pdfs"
base_path = 'krishna'
input_folder = os.path.join(base_path, folder_name)
input_folder = f"{base_path}/{folder_name}"
output_base = f"{base_path}/output"
# pdf_2_png_output_dir = f"{output_base}/pdf_2_png_output_dir"
output_json_folder = f"{output_base}/json"
output_md_folder = f"{output_base}/markdown"
output_excel_folder = f"{output_base}/excel"
# llm_stats_excel = f"{base_path}/llm_processing_stats.xlsx"


# Check and create output directories if they don't exist
for folder in [output_json_folder, output_md_folder, output_excel_folder]:
    os.makedirs(folder, exist_ok=True)

# --- Initialize converter  ---
options = PdfPipelineOptions(
    device=device,
    do_ocr=False,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE
    )
)


# --- Initialize converter  ---
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    }
)


# converter = DocumentConverter()
# --- Loop through all PDFs ---
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(".pdf"):
        continue


    pdf_name = os.path.splitext(file_name)[0]
    source = os.path.join(input_folder, file_name)

    output_json_original = os.path.join(output_json_folder, f"{pdf_name}_original.json")
    output_json_pymupdf = os.path.join(output_json_folder, f"{pdf_name}_pdfminer_metadata.json")
    output_json_hierarchical = os.path.join(output_json_folder, f"{pdf_name}_hierarchical.json")
    # output_json_hierarchical_enriched = os.path.join(output_json_folder, f"{pdf_name}_hierarchical_enriched.json")
    # output_json_yolo_enriched = os.path.join(output_json_folder, f"{pdf_name}_yolo_enriched.json")
    output_json_modified = os.path.join(output_json_folder, f"{pdf_name}_mod.json")

    output_md = os.path.join(output_md_folder, f"{pdf_name}_docling.md")
    output_excel = os.path.join(output_excel_folder, f"{pdf_name}_docling.xlsx")

    print(f"\n Converting document: {source}")

    try:

        # # -----------------------------------------
        # # Pdfminer meta extraction
        # # -----------------------------------------
        # pdfminer_json = extract_lines_with_style(source)
        #
        # with open(output_json_pymupdf, "w", encoding="utf-8") as f:
        #     json.dump(pdfminer_json, f, indent=2, ensure_ascii=False)
        # print(f"PyMuPDF metadata saved: {output_json_pymupdf}")


        # ==================================================
        # CONVERT (RAW DOCLING OUTPUT)
        # ==================================================
        result = converter.convert(source)
        # import pymupdf
        #
        # doc = pymupdf.open(source)
        # print(len(doc.get_toc()))
        # doc.close()
        original_dict = result.document.export_to_dict()

        with open(output_json_original, "w", encoding="utf-8") as f:
            json.dump(original_dict, f, ensure_ascii=False, indent=2)

        print(f"Original JSON saved: {output_json_original}")


        # ==================================================
        # POSTPROCESS (FONT-AWARE HIERARCHY)
        # ==================================================
        ResultPostprocessor(result).process()
        # ResultPostprocessor(result, source=source).process()

        hierarchical_dict = result.document.export_to_dict()

        with open(output_json_hierarchical, "w", encoding="utf-8") as f:
            json.dump(hierarchical_dict, f, ensure_ascii=False, indent=2)

        print(f"Hierarchical JSON saved: {output_json_hierarchical}")


        # # -----------------------------------------
        # # ENRICH (DOCLING JSON ENRICHMENT)
        # # -----------------------------------------
        # docling_json_enriched= attach_pdfminer_styles_to_docling(hierarchical_dict, pdfminer_json)
        #
        # with open(output_json_hierarchical_enriched, "w", encoding="utf-8") as f:
        #     json.dump(docling_json_enriched, f, ensure_ascii=False, indent=2)
        # print(f"Docling JSON enriched: {output_json_hierarchical_enriched}")

        # # ==================================================
        # # YOLO HEADER DETECTION
        # # ==================================================
        # print(f"Running YOLO header detection for {pdf_name}...")
        # yolo_headers_extraction = main(source, pdf_2_png_output_dir)
        # print(yolo_headers_extraction)

        # # -----------------------------------------
        # # YOLO HEADER REFINEMENT (NEW STEP)
        # # -----------------------------------------
        # docling_yolo_json_enriched = promote_headers_from_yolo(
        #     docling_json_enriched,
        #     yolo_headers_extraction,  # from YOLO + OCR
        #     min_conf=0.6
        # )

        # with open(output_json_yolo_enriched, "w", encoding="utf-8") as f:
        #     json.dump(docling_yolo_json_enriched, f, ensure_ascii=False, indent=2)

        # # ---
        # # LLM POST-PROCESSING
        # # ---
        # print(f"Running LLM header correction for {pdf_name}...")
        # docling_json_llm_corrected = llm_header_preprocessing.process_headers_with_llm(
        #     docling_json_enriched,
        #     excel_path=llm_stats_excel,
        #     pdf_name=pdf_name
        # )
        #
        # output_json_llm = os.path.join(output_json_folder, f"{pdf_name}_llm_corrected.json")
        # with open(output_json_llm, "w", encoding="utf-8") as f:
        #     json.dump(docling_json_llm_corrected, f, ensure_ascii=False, indent=2)
        # print(f"LLM Corrected JSON saved: {output_json_llm}")

        # ==================================================
        # ANNOTATE (FINAL MODIFIED STRUCTURE)
        # ==================================================
        # annotator = HierarchyAnnotator(docling_json_llm_corrected)

        annotator = HierarchyAnnotator(hierarchical_dict, source_name=pdf_name)
        annotated_doc = annotator.annotate(export_excel_path=output_excel)

        with open(output_json_modified, "w", encoding="utf-8") as f:
            json.dump(annotated_doc, f, ensure_ascii=False, indent=2)

        print(f"Modified JSON saved: {output_json_modified}")

        # # -----------------------------------------
        # # Post-process header hierarchy in Excel
        # # -----------------------------------------
        # df = process_h_columns(output_excel)
        # print(f"Header hierarchy normalized in Excel: {output_excel}")
        #
        # # -----------------------------------------
        # # FRAGMENT CORRUPTION CLEANING
        # # -----------------------------------------
        # df = fragment_main(df)
        # print(f"Fragment hierarchy normalized in Excel: {output_excel}")
        # df.to_excel(output_excel, index=False)

        # Save Markdown
        markdown_text = DocProcessor(annotated_doc).generate()
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"Markdown saved: {output_md}")

        # # Save Excel
        # print(f"Excel saved: {output_excel}")
        print(f"Final cleaned Excel saved: {output_excel}")

    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

print("\nAll files processed successfully!")



# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
# from hierarchical.postprocessor import ResultPostprocessor
# from hierarchy_annotator_main import HierarchyAnnotator
# from doc_processor import DocProcessor
#
# import os
# import json
# import torch
#
# # =====================================================
# # DEVICE CONFIGURATION
# # =====================================================
# if torch.cuda.is_available():
#     device = "cuda"
#     print("Using CUDA for processing")
# else:
#     device = "cpu"
#     print("CUDA not available. Falling back to CPU")
#
#
# # =====================================================
# # BASE DIRECTORY (CLIENT ROOT)
# # =====================================================
# BASE_DIR = "/Users/krishnavelama/Library/CloudStorage/OneDrive-4CRisk.ai/All_4crisk/Parser/Internal_testing/common_issues/pdfs/client copy"
#
#
# # =====================================================
# # DOCLING CONFIG
# # =====================================================
# options = PdfPipelineOptions(
#     device=device,
#     do_ocr=False,
#     do_table_structure=True,
#     table_structure_options=TableStructureOptions(
#         do_cell_matching=True,
#         mode="accurate"
#     )
# )
#
# converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(pipeline_options=options)
#     }
# )
#
#
# # =====================================================
# # PROCESS CLIENT DIRECTORY
# # =====================================================
# def process_client(client_path):
#
#     client_name = os.path.basename(client_path)
#
#     pdf_folder = os.path.join(client_path, "pdfs")
#
#     if not os.path.exists(pdf_folder):
#         return
#
#     # -----------------------------
#     # CREATE DOCLING FOLDER
#     # -----------------------------
#     docling_base = os.path.join(client_path, "docling")
#
#     json_folder = os.path.join(docling_base, "json")
#     md_folder = os.path.join(docling_base, "markdown")
#     excel_folder = os.path.join(docling_base, "excel")
#
#     for folder in [json_folder, md_folder, excel_folder]:
#         os.makedirs(folder, exist_ok=True)
#
#     print("\n====================================")
#     print(f"CLIENT: {client_name}")
#     print(f"PDF FOLDER: {pdf_folder}")
#     print("====================================")
#
#     # -----------------------------
#     # LOOP PDFs
#     # -----------------------------
#     for file_name in os.listdir(pdf_folder):
#
#         if not file_name.lower().endswith(".pdf"):
#             continue
#
#         pdf_name = os.path.splitext(file_name)[0]
#
#         source = os.path.join(pdf_folder, file_name)
#
#         output_json_original = os.path.join(json_folder, f"{pdf_name}_original.json")
#         output_json_hierarchical = os.path.join(json_folder, f"{pdf_name}_hierarchical.json")
#         output_json_modified = os.path.join(json_folder, f"{pdf_name}_mod.json")
#
#         output_md = os.path.join(md_folder, f"{pdf_name}_docling.md")
#         output_excel = os.path.join(excel_folder, f"{pdf_name}_docling.xlsx")
#
#         print(f"\nProcessing PDF → {file_name}")
#
#         try:
#
#             # ==========================================
#             # DOCLING CONVERSION
#             # ==========================================
#             result = converter.convert(source)
#
#             original_dict = result.document.export_to_dict()
#
#             with open(output_json_original, "w", encoding="utf-8") as f:
#                 json.dump(original_dict, f, ensure_ascii=False, indent=2)
#
#             print(f"Saved → {output_json_original}")
#
#             # ==========================================
#             # POST PROCESSOR
#             # ==========================================
#             ResultPostprocessor(result).process()
#
#             hierarchical_dict = result.document.export_to_dict()
#
#             with open(output_json_hierarchical, "w", encoding="utf-8") as f:
#                 json.dump(hierarchical_dict, f, ensure_ascii=False, indent=2)
#
#             print(f"Saved → {output_json_hierarchical}")
#
#             # ==========================================
#             # HIERARCHY ANNOTATION
#             # ==========================================
#             annotator = HierarchyAnnotator(
#                 hierarchical_dict,
#                 source_name=pdf_name
#             )
#
#             annotated_doc = annotator.annotate(
#                 export_excel_path=output_excel
#             )
#
#             with open(output_json_modified, "w", encoding="utf-8") as f:
#                 json.dump(annotated_doc, f, ensure_ascii=False, indent=2)
#
#             print(f"Saved → {output_json_modified}")
#
#             # ==========================================
#             # MARKDOWN GENERATION
#             # ==========================================
#             markdown_text = DocProcessor(annotated_doc).generate()
#
#             with open(output_md, "w", encoding="utf-8") as f:
#                 f.write(markdown_text)
#
#             print(f"Saved → {output_md}")
#             print(f"Saved → {output_excel}")
#
#         except Exception as e:
#
#             print(f"Failed → {file_name}")
#             print(e)
#
#     print(f"\nCLIENT COMPLETED → {client_name}")
#
#
# # =====================================================
# # MAIN LOOP
# # =====================================================
# for root, dirs, files in os.walk(BASE_DIR):
#
#     if "pdfs" in dirs:
#         process_client(root)
#
# print("\n✔ ALL CLIENTS PROCESSED")