import json
import requests
from bs4 import BeautifulSoup
import logging
import re
from langchain_ollama.chat_models import ChatOllama

# Set up logging
logging.basicConfig(level=logging.INFO)

class MarketResearchAgent:
    def gather_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to retrieve content from {url}: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        products = []

        for product in soup.select("div.product-tuple-listing"):
            product_name = product.select_one("p.product-title").text.strip() if product.select_one("p.product-title") else "No Name"
            product_price = product.select_one("span.lfloat.product-price").text.strip() if product.select_one("span.lfloat.product-price") else "Price Not Available"
            product_ratings = product.select_one("p.product-rating-count").text.strip() if product.select_one("p.product-rating-count") else "Ratings Not Available"
            product_link = product.select_one("a.dp-widget-link")['href'] if product.select_one("a.dp-widget-link") else "Link Not Available"
            image_tag = product.select_one("img")
            
            product_image = image_tag['src'] if image_tag and 'src' in image_tag.attrs else \
                            image_tag['data-src'] if image_tag and 'data-src' in image_tag.attrs else \
                            "Image Not Available"

            products.append({
                "name": product_name,
                "price": product_price,
                "ratings": product_ratings,
                "product_link": product_link,
                "image": product_image
            })

        logging.info(f"Retrieved {len(products)} products from {url}")
        return products


class UseCaseGenerationAgent:
    def __init__(self, chat_api):
        self.chat_api = chat_api

    def generate_use_cases(self):
        prompt = "Generate detailed use cases for AI in e-commerce."
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = self.chat_api.invoke(messages)
            #logging.info(f"API response for use cases: {response}")
            splitted_content= response.content
            logging.info(f" splitted string {splitted_content}")

            # Extract use cases from response
            if response :
                logging.info(f" inside usecases if condition")
                content = splitted_content
                
                # Split content into use cases based on the provided format
                use_cases = re.split(r'\n\n\*\*\d+\.\s', content)
                logging.info(f" after splitting with regex {use_cases}")
                
                use_cases = [{"use_case": uc.strip()} for uc in use_cases if uc.strip()]
                logging.info(f" after for {use_cases}")
                logging.info(f" aafter for loop")  
                
                # Format the use cases to include more details
                formatted_use_cases = []
                count =0
                for uc in use_cases[1:4]:
                    title = uc['use_case'].split("\*\*\n\n\* Use Case")[0]
                    logging.info(f" title : {title}")
                    formatted_use_cases.append({
                        "title": title.strip()
                    })
                
                logging.info(f"Generated {len(formatted_use_cases)} use cases.")
                return formatted_use_cases
            else:
                logging.warning("No content returned from use case generation.")
                return []
        except Exception as e:
            logging.error(f"Error generating use cases: {e}")
            return []


class DatasetSuggestionAgent:
    def __init__(self, chat_api):
        self.chat_api = chat_api

    def suggest_datasets(self, use_cases):
        datasets = {}
        for case in use_cases:
            prompt = f"Suggest datasets links for HuggingFace and Kaggle for the use case: {case['title']}"
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                response = self.chat_api.invoke(messages)
                logging.info(f"API response for datasets: {response}")

                # Extract datasets from response
                if response :
                    logging.info(f" inside dataset if condition: {response}")
                    content = response.content
                    logging.info(f" dataset content: {content}")
                    datasets[case['title']] = [dataset.strip() for dataset in content.split(',')]  
                else:
                    logging.warning(f"No content returned for use case: {case['title']}")
            except Exception as e:
                logging.error(f"Error suggesting datasets for {case['title']}: {e}")

        return datasets


class IntegrationReportingAgent:
    def generate_report(self, use_cases, datasets, products):
        report = {
            "scraped_products": products,
            "use_cases": use_cases,
            "relevant_datasets": datasets
        }
        
        with open('report.json', 'w') as json_file:
            json.dump(report, json_file, indent=4)
        
        logging.info("Report saved to 'report.json'.")


def main():
    snapdeal_url = "https://www.snapdeal.com/products/electronics-home-audio-systems?sort=plrty&page=1"

    # Market Research Agent
    market_agent = MarketResearchAgent()
    products = market_agent.gather_data(snapdeal_url)

    # ChatOllama API setup
    chat_api = ChatOllama(base_url='https://ollama.********.com/', model='llama3')

    # Use Case Generation Agent
    use_case_agent = UseCaseGenerationAgent(chat_api)
    use_cases = use_case_agent.generate_use_cases()

    # Check if use cases were generated
    # if not use_cases:
    #     logging.error("No use cases generated. Exiting.")
    #     return

    # Dataset Suggestion Agent
    dataset_suggestion_agent = DatasetSuggestionAgent(chat_api)
    suggested_datasets = dataset_suggestion_agent.suggest_datasets(use_cases)

    # Integration and Reporting Agent
    reporting_agent = IntegrationReportingAgent()
    reporting_agent.generate_report(use_cases, suggested_datasets, products)


if __name__ == "__main__":
    main()
