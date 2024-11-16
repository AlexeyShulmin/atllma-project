import requests
from bs4 import BeautifulSoup
import time
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from tqdm.autonotebook import tqdm
from llm import get_llm_response

client = chromadb.PersistentClient(path="./chromadb_client_data")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection("doc_search_v1", embedding_function=default_ef)


def scrape_text_from_link(url):
    try:
        # Send a GET request
        headers = {'User-Agent': 'My Web Scraper 1.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as err:
        print(f"Request Exception for {url}: {err}")
        return None

    try:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get the text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        content = '\n'.join(chunk for chunk in chunks if chunk)

        content = content.split('\n')[1091:]
        
        return '\n'.join(content)
    except Exception as e:
        print(f"Parsing Exception for {url}: {e}")
        return None

def parse_links(list_of_links):
    pbar = tqdm(list_of_links, total=len(list_of_links))
    for link in pbar:
        pbar.set_postfix({'link': link})
        text_content = scrape_text_from_link(link)
        context = 'You will recieve a text from a webpage of python library documentation. Describe, what given text is about, focus on use case, drop data about parameters, features, and returns.'
        text_content = get_llm_response(text_content, context)
        if text_content:
            collection.add(
                documents=[text_content],
                ids=[link],
                embeddings=default_ef([text_content])
            )
        time.sleep(1)  # Wait a second before the next request


list_of_links = [
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot_date.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.step.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.loglog.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.semilogx.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.semilogy.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_betweenx.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.eventplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stackplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.broken_barh.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.vlines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hlines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhspan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axline.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.acorr.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.angle_spectrum.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.cohere.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.csd.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.magnitude_spectrum.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.phase_spectrum.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.psd.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.specgram.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.xcorr.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ecdf.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bxp.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist2d.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stairs.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.clabel.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contour.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contourf.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolorfast.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.spy.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tripcolor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.triplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontour.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontourf.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.table.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.arrow.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.indicate_inset.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.indicate_inset_zoom.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.secondary_xaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.secondary_yaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barbs.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.streamplot.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.cla.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.clear.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_axis_off.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_axis_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_frame_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_frame_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_axisbelow.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_axisbelow.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_facecolor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_facecolor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_prop_cycle.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.invert_xaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.xaxis_inverted.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.invert_yaxis.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.yaxis_inverted.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xlim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ylim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.update_datalim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xbound.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xbound.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ybound.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ybound.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xlabel.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ylabel.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.label_outer.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_title.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_legend.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_legend_handles_labels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xscale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yscale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.use_sticky_edges.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.margins.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xmargin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ymargin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xmargin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ymargin.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.relim.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.autoscale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.autoscale_view.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_autoscale_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_autoscale_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_autoscalex_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_autoscalex_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_autoscaley_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_autoscaley_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.apply_aspect.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_aspect.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_box_aspect.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_box_aspect.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_adjustable.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_adjustable.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xticks.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xmajorticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xminorticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xgridlines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xticklines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.xaxis_date.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticks.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yticks.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ymajorticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yminorticklabels.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_ygridlines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yticklines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.yaxis_date.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.minorticks_off.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.minorticks_on.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.locator_params.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.convert_xunits.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.convert_yunits.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.have_units.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_artist.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_child_axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_collection.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_container.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_image.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_line.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_patch.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_table.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twiny.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.sharex.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.sharey.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_shared_x_axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_shared_y_axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_anchor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_anchor.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_axes_locator.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_axes_locator.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_subplotspec.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_subplotspec.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.reset_position.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_position.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_position.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stale.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pchanged.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.add_callback.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.remove_callback.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.can_pan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.can_zoom.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_navigate.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_navigate.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_navigate_mode.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_navigate_mode.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_forward_navigation_events.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_forward_navigation_events.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.start_pan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.drag_pan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.end_pan.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.format_coord.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.format_cursor_data.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.format_xdata.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.format_ydata.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.mouseover.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.in_axes.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contains.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contains_point.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_cursor_data.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_children.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_images.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_lines.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.findobj.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.draw.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.draw_artist.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.redraw_in_frame.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_rasterization_zorder.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_rasterization_zorder.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_window_extent.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_tightbbox.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.name.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xaxis_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yaxis_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_data_ratio.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xaxis_text1_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_xaxis_text2_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yaxis_text1_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_yaxis_text2_transform.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.zorder.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_default_bbox_extra_artists.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.get_transformed_clip_path_and_affine.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.has_data.html',
    'https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html',
]

parse_links(list_of_links)