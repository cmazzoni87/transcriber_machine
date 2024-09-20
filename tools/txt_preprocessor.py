import subprocess
import os
# from tools import DATA_DIR
# import markdown
# from weasyprint import HTML


def convert_markdown_to_pdf(mark_downfile_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using the mdpdf command line application.

    Args:
        mark_downfile_name (str): Path to the input Markdown file.

    Returns:
        str: Path to the generated PDF file.
    """
    output_file = os.path.splitext(mark_downfile_name)[0] + '.pdf'

    # Command to convert markdown to PDF using mdpdf
    cmd = ['mdpdf', '--output', output_file, mark_downfile_name]

    # Execute the command
    subprocess.run(cmd, check=True)
    return output_file


def markdown_to_pdf(markdown_text: str):
    """
    Converts a markdown text to a PDF file, including images.

    :param markdown_text: The input markdown text as a string.
    """
    # Convert markdown to HTML
    # html_content = markdown.markdown(markdown_text)
    # # Inject CSS to resize images
    # css = """
    # <style>
    #     img {
    #         max-width: 100%;
    #         height: auto;
    #         max-height: 500px;
    #     }
    # </style>
    # """
    # # Add CSS to the HTML content
    # html_content = css + html_content
    # # save the html content to a file
    # with open(os.path.join(DATA_DIR, 'output.html'), 'w') as file:
    #     file.write(html_content)
    #
    # # Define the output path
    # output_path = os.path.join(DATA_DIR, 'output.pdf')
    #
    # # Convert HTML to PDF using WeasyPrint
    # HTML(string=html_content, base_url=DATA_DIR).write_pdf(output_path)
    # print(f"PDF generated and saved to {output_path}")
    output_path = "WIP"
    return output_path


def json_to_markdown(data: dict) -> str:
    markdown = []

    # Conversation Summary
    if "conversation_summary" in data:
        markdown.append("## Conversation Summary\n")
        summary = data["conversation_summary"]
        markdown.append(f"- **{summary['topic']}:** {summary['summary']}")
        markdown.append("")

    # Action Items
    if "action_items" in data:
        markdown.append("## Action Items\n")
        for item in data["action_items"]:
            markdown.append(f"- **Description:** {item['description']}")
            markdown.append(f"  - **Responsible Party:** {item['responsible_party']}")
            if item['deadline']:
                markdown.append(f"  - **Deadline:** {item['deadline']}")
            if item['additional_notes']:
                markdown.append(f"  - **Additional Notes:** {item['additional_notes']}")
            markdown.append("")

    # Sentiment Analysis
    if "sentiment_analysis" in data:
        markdown.append("## Sentiment Analysis\n")
        markdown.append(f"- **Overall Sentiment:** {data['sentiment_analysis']['overall_sentiment']}\n")
        for sentiment in data['sentiment_analysis']['detailed_sentiment']:
            markdown.append(f"- **{sentiment['speaker']}** ({sentiment['sentiment']}): {sentiment['remarks']}")
        markdown.append("")

    # Potential Priorities
    if "potential_priorities" in data:
        markdown.append("## Potential Priorities\n")
        for priority in data["potential_priorities"]:
            markdown.append(f"- **Priority Level:** {priority['priority_level']}")
            markdown.append(f"  - **Description:** {priority['description']}")
            if "related_action_items" in priority and priority["related_action_items"]:
                markdown.append(f"  - **Related Action Items:**")
                for action_item in priority["related_action_items"]:
                    markdown.append(f"    - {action_item}")
            markdown.append(f"  - **Strategic Importance:** {priority['strategic_importance']}")
            markdown.append("")

    return "\n".join(markdown)


